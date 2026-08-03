"""ComfyUI node: MiniMax H3 NestedTensor latent upscale + CONST re-noise."""

from __future__ import annotations

from .utils import UPSCALE_METHODS, upscale_and_add_noise


class MiniMaxH3LatentUpscaleCombined:
    """Upscale NestedTensor video latents, then re-noise for DisableNoise continuation."""

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
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale_noise"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Upscale MiniMax H3 NestedTensor video (H, W), re-noise video only for CONST/flow "
        "DisableNoise continuation; leave audio clean (not remixed). "
        "Feed denoised_output from sampler #1; use DisableNoise on sampler #2 with the same sigmas. "
        "Do not put Easy-Use Empty Cache / forced unload between samplers."
    )

    def upscale_noise(self, samples, scale_by, method, model, noise, sigmas):
        return (upscale_and_add_noise(samples, scale_by, method, model, noise, sigmas),)
