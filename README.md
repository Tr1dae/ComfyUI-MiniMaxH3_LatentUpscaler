# MiniMaxH3_LatentUpscaler

ComfyUI custom node for **latent spatial upscaling** between MiniMax H3 samplers.

Not a learned AI upscaler. Stock `LatentUpscaleBy` / `AddNoise` break on MiniMax’s ComfyUI `NestedTensor` AV latents (`video [B,24,T,H/16,W/16]` + `audio [B,32,2,T_audio]`).

## Node

**MiniMax H3 Latent Upscale Combined** (`latent/minimax_h3`)

Inputs: `LATENT`, `scale_by`, `method` (`nearest` | `bilinear` | `bicubic`), `MODEL`, `NOISE`, `SIGMAS`  
Output: `LATENT` (video spatially upscaled; audio unchanged; CONST-ready re-noise)

Does:

1. Upscale NestedTensor video `H,W` via `F.interpolate`
2. Re-noise at `sigmas[0]` with `noise_scaling` + `inverse_noise_scaling` so **DisableNoise** on sampler #2 reconstitutes correctly under CONST/flow

## Wiring

1. SamplerCustomAdvanced #1 → high σ half (`SplitSigmas`)
2. Take **`denoised_output`** (not `output` — upscaling noisy `output` → colored dots)
3. **MiniMax H3 Latent Upscale Combined** — RandomNoise + low sigmas + same model
4. SamplerCustomAdvanced #2 — **DisableNoise** + same low sigmas + combined output

Avoid Easy-Use Empty Cache / forced model unload between the two samplers (can AV-crash quantized MiniMax).

## Install

`ComfyUI/custom_nodes/MiniMaxH3_LatentUpscaler/` — restart or reload custom nodes.
