"""MiniMax H3 NestedTensor latent helpers: extract, wrap, upscale, CONST re-noise."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

import comfy.nested_tensor

UPSCALE_METHODS = ("nearest", "bilinear", "bicubic")

# MiniMax DiT patch_size is (1, 2, 2); cond patchify does not pad, so H/W must be even.
_SPATIAL_MULTIPLE = 2


def _snap_spatial(size: int, multiple: int = _SPATIAL_MULTIPLE) -> int:
    return max(multiple, ((int(size) + multiple - 1) // multiple) * multiple)


def is_nested_tensor(obj: Any) -> bool:
    return isinstance(obj, comfy.nested_tensor.NestedTensor) or getattr(obj, "is_nested", False)


def extract_tensor(samples: Any) -> tuple[list[torch.Tensor], bool]:
    """Unwrap NestedTensor members, or wrap a plain torch.Tensor in a one-element list.

    Returns (tensors, was_nested).
    """
    if is_nested_tensor(samples):
        tensors = list(samples.unbind())
        if not tensors:
            raise ValueError("NestedTensor has an empty .tensors list")
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"NestedTensor member [{i}] is {type(t)}, expected torch.Tensor")
        return tensors, True
    if isinstance(samples, torch.Tensor):
        return [samples], False
    raise TypeError(f"Expected NestedTensor or torch.Tensor, got {type(samples)}")


def wrap_tensor(
    tensors: Sequence[torch.Tensor],
    *,
    was_nested: bool,
) -> Any:
    """Rebuild NestedTensor with ComfyUI's constructor, or return a single plain tensor."""
    if was_nested:
        if not tensors:
            raise ValueError("Cannot wrap empty tensor list as NestedTensor")
        return comfy.nested_tensor.NestedTensor(tensors)
    if len(tensors) != 1:
        raise ValueError(f"Plain latent path expects one tensor, got {len(tensors)}")
    return tensors[0]


def upscale_video_latent(
    video: torch.Tensor,
    scale_by: float,
    method: str,
) -> torch.Tensor:
    """Spatially upscale a video latent [B, C, T, H, W] (or [B, C, H, W]).

    Mirrors comfy.utils.common_upscale's >4D path: fold intermediate dims into batch,
    interpolate H/W, then restore.
    """
    if method not in UPSCALE_METHODS:
        raise ValueError(f"Unsupported method {method!r}; expected one of {UPSCALE_METHODS}")
    if video.ndim < 4:
        raise ValueError(f"Video latent needs at least 4 dims [B,C,H,W], got shape {tuple(video.shape)}")

    height = _snap_spatial(max(1, round(video.shape[-2] * scale_by)))
    width = _snap_spatial(max(1, round(video.shape[-1] * scale_by)))

    orig_shape = tuple(video.shape)
    samples = video
    if len(orig_shape) > 4:
        # [B, C, T, H, W] -> [B*T, C, H, W]
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])

    if method in ("bilinear", "bicubic"):
        out = F.interpolate(samples, size=(height, width), mode=method, align_corners=False)
    else:
        out = F.interpolate(samples, size=(height, width), mode="nearest")

    if len(orig_shape) == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


def upscale_nested_latent(
    latent: dict,
    scale_by: float,
    method: str,
) -> dict:
    """Copy LATENT dict; upscale video spatial dims; pass audio through; rebuild NestedTensor."""
    if "samples" not in latent:
        raise KeyError('LATENT dict missing "samples"')

    out = latent.copy()
    members, was_nested = extract_tensor(latent["samples"])

    if was_nested:
        # MiniMax H3 / AV: tensors[0]=video [B,24,T,H,W], tensors[1]=audio [B,32,2,Ta]
        video_up = upscale_video_latent(members[0], scale_by, method)
        out["samples"] = wrap_tensor([video_up, *members[1:]], was_nested=True)

        noise_mask = latent.get("noise_mask")
        if noise_mask is not None and is_nested_tensor(noise_mask):
            mask_members, _ = extract_tensor(noise_mask)
            if not mask_members:
                raise ValueError("noise_mask NestedTensor is empty")
            video_mask_up = upscale_video_latent(mask_members[0], scale_by, method)
            out["noise_mask"] = wrap_tensor([video_mask_up, *mask_members[1:]], was_nested=True)
        elif noise_mask is not None and isinstance(noise_mask, torch.Tensor) and noise_mask.ndim >= 4:
            out["noise_mask"] = upscale_video_latent(noise_mask, scale_by, method)
    else:
        out["samples"] = upscale_video_latent(members[0], scale_by, method)
        noise_mask = latent.get("noise_mask")
        if noise_mask is not None and isinstance(noise_mask, torch.Tensor) and noise_mask.ndim >= 4:
            out["noise_mask"] = upscale_video_latent(noise_mask, scale_by, method)

    return out


def _has_nonzero(samples: Any) -> bool:
    """NestedTensor-safe replacement for torch.count_nonzero(samples) > 0."""
    members, _ = extract_tensor(samples)
    return any(torch.count_nonzero(t) > 0 for t in members)


def add_noise_nested_latent(
    model: Any,
    noise: Any,
    sigmas: torch.Tensor,
    latent: dict,
    *,
    renoise_indices: Sequence[int] | None = None,
    noise_strengths: dict[int, float] | None = None,
) -> dict:
    """NestedTensor-aware AddNoise for CONST/flow DisableNoise continuation.

    Mixes at sigmas[0], then inverse_noise_scaling(sigmas[0]) so SamplerCustomAdvanced
    + DisableNoise reconstitutes the intended noisy latent (CONST starts as (1-σ)·latent).

    renoise_indices: members that get fresh noise mix. Default: all.
    Members not in the set stay clean but are still inverse-scaled so DisableNoise
    reconstitutes clean (not (1-σ)·clean).

    noise_strengths: optional per-member multiplier on the noise tensor before
    noise_scaling (e.g. {1: 0.5} half-strength audio remix). Missing keys = 1.0.
    """
    if len(sigmas) == 0:
        return latent

    if "samples" not in latent:
        raise KeyError('LATENT dict missing "samples"')

    out = latent.copy()
    latent_image = latent["samples"]
    noisy = noise.generate_noise(latent)

    model_sampling = model.get_model_object("model_sampling")
    process_latent_out = model.get_model_object("process_latent_out")
    process_latent_in = model.get_model_object("process_latent_in")

    sigma_start = sigmas[0]

    lat_members, was_nested = extract_tensor(latent_image)
    noise_members, noise_was_nested = extract_tensor(noisy)
    if len(lat_members) != len(noise_members):
        raise ValueError(
            f"Noise NestedTensor has {len(noise_members)} members but latent has {len(lat_members)}"
        )

    if renoise_indices is None:
        renoise_set = set(range(len(lat_members)))
    else:
        renoise_set = set(renoise_indices)
        for i in renoise_set:
            if i < 0 or i >= len(lat_members):
                raise IndexError(f"renoise index {i} out of range for {len(lat_members)} members")

    shift_latents = _has_nonzero(latent_image)

    result_members: list[torch.Tensor] = []
    for i, (lat, noi) in enumerate(zip(lat_members, noise_members)):
        lat_i = process_latent_in(lat) if shift_latents else lat
        if i in renoise_set:
            strength = 1.0
            if noise_strengths is not None:
                strength = float(noise_strengths.get(i, 1.0))
            strength = max(0.0, min(1.0, strength))
            if strength <= 0.0:
                mixed = lat_i
            elif strength >= 1.0:
                mixed = model_sampling.noise_scaling(sigma_start, noi, lat_i)
            else:
                # Reduce noise amplitude -> lighter denoise on this stream.
                mixed = model_sampling.noise_scaling(sigma_start, noi * strength, lat_i)
        else:
            mixed = lat_i
        if hasattr(model_sampling, "inverse_noise_scaling"):
            mixed = model_sampling.inverse_noise_scaling(sigma_start, mixed)
        mixed = process_latent_out(mixed)
        mixed = torch.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)
        result_members.append(mixed)

    out["samples"] = wrap_tensor(result_members, was_nested=was_nested or noise_was_nested)
    return out


def _move_samples_to_cpu(samples: Any) -> Any:
    """Detach NestedTensor / Tensor samples onto CPU for cheap cache between samplers."""
    if is_nested_tensor(samples):
        return samples.cpu()
    if isinstance(samples, torch.Tensor):
        return samples.detach().to("cpu")
    return samples


def finalize_latent_for_handoff(latent: dict) -> dict:
    """Park LATENT on CPU and soft-clear CUDA cache without unloading models.

    Forced unload / Easy-Use Empty Cache between MiniMax passes is unsafe with
    quantized weights + --disable-dynamic-vram (ends as 0MB loaded / illegal
    SageAttention access). Soft empty_cache only frees allocator fragments.
    """
    out = latent.copy()
    if "samples" in out:
        out["samples"] = _move_samples_to_cpu(out["samples"])
    if "noise_mask" in out:
        out["noise_mask"] = _move_samples_to_cpu(out["noise_mask"])

    try:
        import comfy.model_management as mm
        import gc

        gc.collect()
        mm.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def upscale_and_add_noise(
    latent: dict,
    scale_by: float,
    method: str,
    model: Any,
    noise: Any,
    sigmas: torch.Tensor,
    *,
    audio_denoise: float = 1.0,
) -> dict:
    """Upscale video spatially; re-noise video fully; optionally re-noise audio.

    audio_denoise: 0 = keep pass-1 audio clean (pass 2 won't rewrite it much),
    1 = full CONST remix of audio at sigmas[0] so pass 2 can improve it,
    values in between scale the audio noise amplitude.
    """
    upscaled = upscale_nested_latent(latent, scale_by, method)
    members, was_nested = extract_tensor(upscaled["samples"])
    if was_nested and len(members) >= 2:
        strength = max(0.0, min(1.0, float(audio_denoise)))
        if strength <= 0.0:
            renoise: tuple[int, ...] | None = (0,)
            strengths = None
        elif strength >= 1.0:
            renoise = (0, 1)
            strengths = None
        else:
            renoise = (0, 1)
            strengths = {0: 1.0, 1: strength}
    else:
        renoise = None
        strengths = None
    noised = add_noise_nested_latent(
        model,
        noise,
        sigmas,
        upscaled,
        renoise_indices=renoise,
        noise_strengths=strengths,
    )
    return finalize_latent_for_handoff(noised)


def _upscale_video_like_latent(z: torch.Tensor, scale_by: float, method: str) -> torch.Tensor:
    """Upscale a MiniMax visual latent; supports 5D video or 4D image-like tensors.

    Snaps H/W to the DiT spatial patch multiple (2). Cond path patchify_video does not
    pad, so odd sizes after 1.5× (e.g. 50→75) crash reshape.
    """
    if not isinstance(z, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for visual latent, got {type(z)}")
    if z.ndim == 4:
        z5 = upscale_video_latent(z.unsqueeze(2), scale_by, method)
        # pad T,H,W with MiniMax patch (1,2,2)
        import comfy.ldm.common_dit as common_dit
        z5 = common_dit.pad_to_patch_size(z5, (1, _SPATIAL_MULTIPLE, _SPATIAL_MULTIPLE))
        return z5.squeeze(2)
    if z.ndim == 5:
        z5 = upscale_video_latent(z, scale_by, method)
        import comfy.ldm.common_dit as common_dit
        return common_dit.pad_to_patch_size(z5, (1, _SPATIAL_MULTIPLE, _SPATIAL_MULTIPLE))
    raise ValueError(f"Visual latent needs 4 or 5 dims, got shape {tuple(z.shape)}")


def upscale_minimax_ref_block(block: dict, scale_by: float, method: str) -> dict:
    """Spatially upscale one minimax_refs block; leave audio-only fields unchanged."""
    out = dict(block)
    kind = out.get("kind")
    if kind == "audio":
        return out

    if "latent" in out and out["latent"] is not None:
        z = _upscale_video_like_latent(out["latent"], scale_by, method)
        out["latent"] = z
        # PackedLayout reads these for RoPE / row counts — keep in sync with tensor.
        if z.ndim == 5:
            out["latent_h"] = int(z.shape[-2])
            out["latent_w"] = int(z.shape[-1])
            if "latent_t" in out:
                out["latent_t"] = int(z.shape[2])
        elif z.ndim == 4:
            out["latent_h"] = int(z.shape[-2])
            out["latent_w"] = int(z.shape[-1])

    # audio_latent / ref_audio_t intentionally untouched (no spatial dims)
    return out


def upscale_minimax_keyframe(kf: dict, scale_by: float, method: str) -> dict:
    """Upscale a minimax_keyframes entry so it matches the new target spatial grid."""
    out = dict(kf)
    if "latent" in out and out["latent"] is not None:
        out["latent"] = _upscale_video_like_latent(out["latent"], scale_by, method)
    return out


def upscale_minimax_conditioning(
    conditioning: list | None,
    scale_by: float,
    method: str,
) -> list | None:
    """Clone CONDITIONING and spatially upscale MiniMax ref/keyframe visual latents.

    Updates minimax_refs[*].latent (+ latent_h/w/t) and minimax_keyframes[*].latent.
    Text / audio cond tensors are left alone. Rebuild the Guider from the returned
    conditioning for sampler #2 (do not reuse a Guider built on pre-upscale cond).
    """
    if conditioning is None:
        return None

    out: list = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            out.append(entry)
            continue
        emb, meta = entry[0], entry[1]
        new_meta = meta.copy()

        refs = meta.get("minimax_refs")
        if refs is not None:
            new_meta["minimax_refs"] = [
                upscale_minimax_ref_block(blk, scale_by, method) for blk in refs
            ]

        keyframes = meta.get("minimax_keyframes")
        if keyframes is not None:
            new_meta["minimax_keyframes"] = [
                upscale_minimax_keyframe(kf, scale_by, method) for kf in keyframes
            ]

        out.append([emb, new_meta])
    return out
