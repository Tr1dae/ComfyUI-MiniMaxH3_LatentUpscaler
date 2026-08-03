"""ComfyUI node: MiniMax H3 NestedTensor latent upscale + CONST re-noise."""

from __future__ import annotations

from .utils import (
    UPSCALE_METHODS,
    upscale_and_add_noise,
    upscale_minimax_conditioning,
)


class MiniMaxH3LatentUpscaleCombined:
    """Upscale NestedTensor video latents + MiniMax ref/keyframe cond, then re-noise."""

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
                "audio_denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "How hard pass 2 may rewrite audio. "
                            "0 = lock pass-1 audio (clean, no remix). "
                            "1 = full re-noise at sigmas[0] so sampler 2 can improve audio. "
                            "Try 0.25–0.5 for light polish."
                        ),
                    },
                ),
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
        "Upscale MiniMax H3 NestedTensor video (H, W) and CONST-prepare streams for DisableNoise. "
        "Video is always re-noised at sigmas[0]. audio_denoise controls whether audio is remixed "
        "so sampler #2 can improve it (1) or stay locked from pass 1 (0). "
        "Optionally upscale minimax_refs / keyframes — rebuild Guider from returned CONDITIONING."
    )

    def upscale_noise(
        self,
        samples,
        scale_by,
        method,
        model,
        noise,
        sigmas,
        audio_denoise=1.0,
        positive=None,
        negative=None,
    ):
        latent = upscale_and_add_noise(
            samples,
            scale_by,
            method,
            model,
            noise,
            sigmas,
            audio_denoise=audio_denoise,
        )
        pos_out = upscale_minimax_conditioning(positive, scale_by, method)
        neg_out = upscale_minimax_conditioning(negative, scale_by, method)
        return (latent, pos_out, neg_out)
