# Wan2.2-Remix Workflows

Wan2.2-Remix is served through SGLang workflow presets. Callers send one HTTP
request with text, or image plus text, and SGLang returns a video artifact.

ComfyUI is not required. SGLang does not execute ComfyUI workflow JSON graphs.
The `*-comfy-v1` presets are SGLang-native translations of the relevant ComfyUI
generation semantics: sampler, schedule, flow shift, step count, high/low
transformer ranges, CFG, resolution, frames, and fps.

## Presets

| Preset | Task | Intent |
|--------|------|--------|
| `wan2.2-remix/nsfw-t2v-comfy-v1` | T2V | Comfy-derived T2V semantics |
| `wan2.2-remix/nsfw-i2v-comfy-v1` | I2V | Comfy-derived I2V semantics |
| `wan2.2-remix/nsfw-t2v-sglang-v1` | T2V | SGLang-native Wan2.2 defaults |
| `wan2.2-remix/nsfw-i2v-sglang-v1` | I2V | SGLang-native Wan2.2 defaults |
| `wan2.2-remix/sfw-t2v-sglang-v1` | T2V | SGLang-native Wan2.2 defaults |

Use the Comfy-derived presets when you want the provided Wan2.2-Remix ComfyUI
workflows' important scalar settings. Use the SGLang presets when you want the
regular Wan2.2 SGLang defaults.

## Start A T2V Server

Serve a converted T2V model tree:

```bash
sglang serve \
  --model-type diffusion \
  --model-path /path/to/Wan2p2_T2V_A14B_Remix_NSFW_v2.0_SGLang \
  --model-id Wan2.2-T2V-A14B-Diffusers \
  --num-gpus 8 \
  --ulysses-degree 4 \
  --ring-degree 2 \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --host 0.0.0.0 \
  --port 30010 \
  --output-path /path/to/outputs \
  --input-save-path /path/to/inputs
```

For lower memory smoke tests, reduce `--num-gpus`, resolution, frame count, or
step count. The full Comfy-derived T2V reference settings are `1280x720`,
`81` frames, and `12` inference steps.

## Start An I2V Server

Serve a converted I2V model tree:

```bash
sglang serve \
  --model-type diffusion \
  --model-path /path/to/Wan2p2_I2V_A14B_Remix_NSFW_v3.0_fp8_SGLang \
  --model-id Wan2.2-I2V-A14B-Diffusers \
  --num-gpus 8 \
  --ulysses-degree 4 \
  --ring-degree 2 \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --host 0.0.0.0 \
  --port 30010 \
  --output-path /path/to/outputs \
  --input-save-path /path/to/inputs
```

The full Comfy-derived I2V reference settings are `1280x720`, `33` frames, and
`12` inference steps.

## List Workflows

```bash
curl -sS "http://localhost:30010/v1/workflows?model_family=wan2.2-remix"
```

Each item includes the preset name, task, defaults, sampler metadata, expert
ranges, and execution status.

## Text To Video

Submit a T2V workflow run:

```bash
curl -sS "http://localhost:30010/v1/workflows/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "wan2.2-remix/nsfw-t2v-comfy-v1",
    "model": "Wan2.2-T2V-A14B-Diffusers",
    "input": {
      "prompt": "a calm cinematic shot of clouds moving over a mountain lake",
      "negative_prompt": "low quality, blurry, artifacts"
    },
    "parameters": {
      "width": 1280,
      "height": 720,
      "num_frames": 81,
      "fps": 16,
      "num_inference_steps": 12,
      "seed": 20260519,
      "guidance_scale": 1.0,
      "guidance_scale_2": 1.0
    },
    "output": {
      "response_format": "url"
    }
  }'
```

The response contains a workflow run ID, the output path, and the effective
parameters. The request returns immediately with `status: queued`.

## Image To Video

Submit an I2V workflow run. `input_reference` may be a local path visible to the
server, a remote URL, or a data image.

```bash
curl -sS "http://localhost:30010/v1/workflows/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "wan2.2-remix/nsfw-i2v-comfy-v1",
    "model": "Wan2.2-I2V-A14B-Diffusers",
    "input": {
      "prompt": "animate the image with slow natural camera motion",
      "negative_prompt": "low quality, blurry, artifacts",
      "input_reference": "/path/to/input.png"
    },
    "parameters": {
      "width": 1280,
      "height": 720,
      "num_frames": 33,
      "fps": 16,
      "num_inference_steps": 12,
      "seed": 20260520,
      "guidance_scale": 1.0,
      "guidance_scale_2": 1.0
    },
    "output": {
      "response_format": "url"
    }
  }'
```

## Poll And Download

Poll a run:

```bash
curl -sS "http://localhost:30010/v1/workflows/runs/<RUN_ID>"
```

When `status` is `completed`, download the video:

```bash
curl -sS -L "http://localhost:30010/v1/workflows/runs/<RUN_ID>/content" \
  -o output.mp4
```

If cloud storage is not configured, the response URL is a relative content URL
served by the same SGLang process. The run response also includes `file_path`
when the server was started with a persistent `--output-path`.

## What Is Guaranteed

Supported:

- one HTTP request produces one video run
- T2V text input and I2V image plus text input
- preset-level overrides for seed, dimensions, frames, fps, steps, guidance, and
  output format fields allowed by the preset
- Comfy-derived Wan2.2-Remix scalar semantics through SGLang-native execution

Not supported:

- uploading arbitrary ComfyUI workflow JSON and executing it
- running ComfyUI nodes in-process
- pixel-level equivalence with ComfyUI outputs
- per-step external control over latent tensors

For static verification of the Comfy-derived preset mapping, see
[Wan2.2-Remix Comfy Workflow Static Alignment](wan2_2_remix_comfy_alignment.md).
