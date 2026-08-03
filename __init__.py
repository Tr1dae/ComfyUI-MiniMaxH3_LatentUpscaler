"""MiniMaxH3_LatentUpscaler — NestedTensor-aware latent inspect / upscale for MiniMax H3."""

from .nodes import (
    MiniMaxH3AddNoise,
    MiniMaxH3LatentInspector,
    MiniMaxH3LatentUpscale,
    MiniMaxH3LatentUpscaleCombined,
)

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LatentInspector": MiniMaxH3LatentInspector,
    "MiniMaxH3LatentUpscale": MiniMaxH3LatentUpscale,
    "MiniMaxH3LatentUpscaleCombined": MiniMaxH3LatentUpscaleCombined,
    "MiniMaxH3AddNoise": MiniMaxH3AddNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LatentInspector": "MiniMax H3 Latent Inspector",
    "MiniMaxH3LatentUpscale": "MiniMax H3 Latent Upscale",
    "MiniMaxH3LatentUpscaleCombined": "MiniMax H3 Latent Upscale Combined",
    "MiniMaxH3AddNoise": "MiniMax H3 Add Noise",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
