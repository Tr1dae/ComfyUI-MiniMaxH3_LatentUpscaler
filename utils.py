"""MiniMax H3 NestedTensor latent helpers: inspect, extract, wrap, upscale."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

import comfy.nested_tensor

# Attr names probed during inspection (Comfy NestedTensor does not define these).
_PROBE_ATTRS = ("tensor", "values", "data", "mask", "storage", "payload")

UPSCALE_METHODS = ("nearest", "bilinear", "bicubic")


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


def _public_attrs(obj: Any) -> list[str]:
    return [name for name in dir(obj) if not name.startswith("_")]


def _probe_attrs(obj: Any) -> dict[str, bool]:
    return {name: hasattr(obj, name) for name in _PROBE_ATTRS}


def _format_tensor(t: torch.Tensor, indent: str = "  ") -> list[str]:
    return [
        f"{indent}type: {type(t)}",
        f"{indent}shape: {tuple(t.shape)}",
        f"{indent}dtype: {t.dtype}",
        f"{indent}device: {t.device}",
        f"{indent}ndim: {t.ndim}",
        f"{indent}requires_grad: {t.requires_grad}",
    ]


def inspect_value(obj: Any, name: str = "value", indent: str = "", depth: int = 0) -> str:
    """Recursively describe a latent field (NestedTensor, Tensor, or other)."""
    pad = indent
    lines: list[str] = [f"{pad}{name}:"]
    inner = indent + "  "

    if is_nested_tensor(obj):
        lines.append(f"{inner}type: {type(obj)}")
        lines.append(f"{inner}is_nested: {getattr(obj, 'is_nested', None)}")
        lines.append(f"{inner}public attrs: {_public_attrs(obj)}")
        lines.append(f"{inner}probed attrs: {_probe_attrs(obj)}")
        try:
            lines.append(f"{inner}shape (first member): {tuple(obj.shape)}")
        except Exception as e:
            lines.append(f"{inner}shape: <error {e}>")
        try:
            lines.append(f"{inner}dtype: {obj.dtype}")
            lines.append(f"{inner}device: {obj.device}")
            lines.append(f"{inner}ndim: {obj.ndim}")
        except Exception as e:
            lines.append(f"{inner}dtype/device/ndim: <error {e}>")

        members = list(getattr(obj, "tensors", obj.unbind()))
        lines.append(f"{inner}member_count: {len(members)}")
        for i, member in enumerate(members):
            role = ""
            if len(members) == 2:
                role = " (video)" if i == 0 else " (audio)"
            lines.append(f"{inner}tensors[{i}]{role}:")
            if isinstance(member, torch.Tensor):
                lines.extend(_format_tensor(member, indent=inner + "  "))
            elif is_nested_tensor(member) and depth < 4:
                lines.append(inspect_value(member, name=f"nested[{i}]", indent=inner + "  ", depth=depth + 1))
            else:
                lines.append(f"{inner}  type: {type(member)} repr={repr(member)[:200]}")
        return "\n".join(lines)

    if isinstance(obj, torch.Tensor):
        lines.extend(_format_tensor(obj, indent=inner))
        lines.append(f"{inner}probed attrs: {_probe_attrs(obj)}")
        return "\n".join(lines)

    if isinstance(obj, dict) and depth < 4:
        lines.append(f"{inner}type: {type(obj)}")
        lines.append(f"{inner}keys: {list(obj.keys())}")
        for k, v in obj.items():
            lines.append(inspect_value(v, name=str(k), indent=inner, depth=depth + 1))
        return "\n".join(lines)

    lines.append(f"{inner}type: {type(obj)}")
    lines.append(f"{inner}repr: {repr(obj)[:300]}")
    return "\n".join(lines)


def inspect_latent(latent: dict) -> str:
    """Full console-oriented report for a ComfyUI LATENT dict."""
    lines = [
        "=== MiniMaxH3 Latent Inspector ===",
        f"type(latent): {type(latent)}",
    ]
    if not isinstance(latent, dict):
        lines.append(f"expected dict, got: {repr(latent)[:300]}")
        return "\n".join(lines)

    lines.append(f"latent.keys(): {list(latent.keys())}")
    for key, value in latent.items():
        lines.append(inspect_value(value, name=f'latent["{key}"]'))
    lines.append("=== end inspector ===")
    return "\n".join(lines)


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

    height = max(1, round(video.shape[-2] * scale_by))
    width = max(1, round(video.shape[-1] * scale_by))

    orig_shape = tuple(video.shape)
    samples = video
    if len(orig_shape) > 4:
        # [B, C, T, H, W] -> [B*T, C, H, W]
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])

    mode = method
    if mode in ("bilinear", "bicubic"):
        out = F.interpolate(samples, size=(height, width), mode=mode, align_corners=False)
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
        # MiniMax H3 / AV pattern: tensors[0]=video [B,24,T,H,W], tensors[1]=audio [B,32,2,Ta]
        video = members[0]
        rest = members[1:]
        video_up = upscale_video_latent(video, scale_by, method)
        out["samples"] = wrap_tensor([video_up, *rest], was_nested=True)

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
