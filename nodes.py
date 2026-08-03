"""ComfyUI node: MiniMax H3 NestedTensor latent upscale + CONST re-noise."""

from __future__ import annotations

from .utils import (
    UPSCALE_METHODS,
    upscale_and_add_noise,
    upscale_minimax_conditioning,
)


class MiniMaxH3LatentUpscaleCombined:
    """Upscale NestedTensor video latents + MiniMax ref/keyframe cond, then re-noise video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),
                "method": (list(UPSCALE_METHODS), {"default": "bilinear"}),
                "model": ("MODEL",),
                "noise": ("NOISE",),
                "sigmas": ("SIGMAS",),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "positive", "negative")
    FUNCTION = "upscale_noise"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Upscale MiniMax H3 NestedTensor video (H, W), re-noise video only for CONST/flow "
        "DisableNoise continuation; leave audio clean. "
        "Optionally upscale minimax_refs / minimax_keyframes visual latents (and latent_h/w) "
        "so reference identity matches the new canvas — rebuild Guider from the returned CONDITIONING "
        "for sampler #2. Do not force-unload VRAM between samplers."
    )

    def upscale_noise(
        self,
        samples,
        scale_by,
        method,
        model,
        noise,
        sigmas,
        positive=None,
        negative=None,
    ):
        latent = upscale_and_add_noise(samples, scale_by, method, model, noise, sigmas)
        pos_out = upscale_minimax_conditioning(positive, scale_by, method)
        neg_out = upscale_minimax_conditioning(negative, scale_by, method)
        return (latent, pos_out, neg_out)
