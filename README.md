# MiniMaxH3_LatentUpscaler

ComfyUI custom node for **latent spatial upscaling** between MiniMax H3 samplers.

Not a learned AI upscaler. Stock `LatentUpscaleBy` / `AddNoise` break on MiniMax’s ComfyUI `NestedTensor` AV latents (`video [B,24,T,H/16,W/16]` + `audio [B,32,2,T_audio]`).

## Node

**MiniMax H3 Latent Upscale Combined** (`latent/minimax_h3`)

**Required inputs:** `LATENT`, `scale_by`, `method`, `MODEL`, `NOISE`, `SIGMAS`  
**Optional inputs:** `positive`, `negative` (`CONDITIONING`)  
**Outputs:** `latent`, `positive`, `negative`

Does:

1. Upscale NestedTensor video `H,W` via `F.interpolate` (audio tensor passed through)
2. Re-noise **video only** at `sigmas[0]` (`noise_scaling` + `inverse_noise_scaling`)
3. Keep **audio clean** (inverse-scaled for DisableNoise)
4. If CONDITIONING is connected: spatially upscale `minimax_refs` / `minimax_keyframes` visual latents and sync `latent_h` / `latent_w` (ref audio left alone)
5. Park LATENT on CPU + `soft_empty_cache` (no model unload)

## Wiring

1. SamplerCustomAdvanced #1 → high/majority σ at low res  
2. Take **`denoised_output`**
3. **MiniMax H3 Latent Upscale Combined**
   - latent = denoised_output  
   - positive/negative = same cond used for pass 1 (ref2va / keyframes)  
   - RandomNoise + low sigmas + model  
4. Build a **new Guider** from Combined’s returned `positive` / `negative` (do **not** reuse the pass-1 Guider)  
5. SamplerCustomAdvanced #2 — DisableNoise + low sigmas + Combined latent + new Guider  

Skipping step 4 leaves ref identity at the old canvas scale → warping / ghosting / seams after 2×.

### Why conditioning must scale (ref2va)

`minimax_refs` packs each ref with its own `latent` + `latent_h`/`latent_w`. After the target canvas grows 2×, refs sized for the 0.5MP “match” canvas sit at the wrong relative scale and RoPE row layout vs the new target — classic identity warp. Combined doubles ref visual latents and metadata together.

### VRAM between samplers

Avoid Easy-Use Empty Cache / force-unload between passes (especially with `--disable-dynamic-vram` + quantized MiniMax + SageAttention).

## Install

`ComfyUI/custom_nodes/MiniMaxH3_LatentUpscaler/` — restart or reload custom nodes.
