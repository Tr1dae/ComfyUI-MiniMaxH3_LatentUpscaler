# MiniMaxH3_LatentUpscaler

ComfyUI custom node for **latent spatial upscaling** between MiniMax H3 samplers.

Not a learned AI upscaler. Stock `LatentUpscaleBy` / `AddNoise` break on MiniMax’s ComfyUI `NestedTensor` AV latents (`video [B,24,T,H/16,W/16]` + `audio [B,32,2,T_audio]`).

## Node

**MiniMax H3 Latent Upscale Combined** (`latent/minimax_h3`)

Inputs: `LATENT`, `scale_by`, `method` (`nearest` | `bilinear` | `bicubic`), `MODEL`, `NOISE`, `SIGMAS`  
Output: `LATENT` (video spatially upscaled + re-noised; **audio shape unchanged and not re-noised**)

Does:

1. Upscale NestedTensor video `H,W` via `F.interpolate` (audio tensor passed through)
2. Re-noise **video only** at `sigmas[0]` (`noise_scaling` + `inverse_noise_scaling`)
3. Keep **audio clean** (inverse-scaled for DisableNoise so pass 2 starts on clean audio, not dampened `(1−σ)·audio`)
4. Park LATENT on CPU + `soft_empty_cache` (no model unload)

## Wiring

1. SamplerCustomAdvanced #1 → high σ half (`SplitSigmas`)
2. Take **`denoised_output`** (not `output` — upscaling noisy `output` → colored dots)
3. **MiniMax H3 Latent Upscale Combined** — RandomNoise + low sigmas + same model
4. SamplerCustomAdvanced #2 — **DisableNoise** + same low sigmas + combined output

### VRAM between samplers (0.5MP → 2× → 1MP)

A clean single-pass 1MP run is not the same as pass1→upscale→pass2:

- Pass 1 leaves MiniMax (and often CLIP/VAE) resident; `--disable-dynamic-vram` worsens reclaim.
- 2× spatial ≈ **4×** video tokens (`H×W`), so pass-2 attention activations jump hard before weights settle.
- Forced Empty Cache / `unload_all_models` mid-graph can leave logs like `0.00 MB usable, ~20GB offloaded`, then SageAttention Triton illegal memory access.

Combined node already parks the LATENT on CPU and calls `soft_empty_cache` only (no model unload).

Do:

1. **No** Easy-Use Empty Cache / force-unload between sampler 1 and 2.
2. Prefer keeping MiniMax loaded across both passes.
3. If pass 2 still spasms: try removing `--disable-dynamic-vram`, or temporarily disable KJNodes MiniMax SageAttention for the high-res pass.
4. Ensure CLIP/VAE aren’t pinned on GPU during pass 2 if you don’t need them yet.

## Install

`ComfyUI/custom_nodes/MiniMaxH3_LatentUpscaler/` — restart or reload custom nodes.
