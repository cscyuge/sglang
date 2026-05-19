# Generic Workflow Execution Service for SGLang Diffusion

## Status

Implementation in progress.

Implemented so far:

- SGLang-native Wan2.2-Remix workflow presets and `/v1/workflows` endpoints.
- Workflow run dispatch through the existing video generation scheduler path.
- Workflow effective-parameter reporting, including sampler and expert metadata.
- Runtime parsing for explicit high/low denoising expert step ranges.
- Fallback to existing `boundary_ratio` behavior when presets do not define
  explicit step ranges.
- Comfy-derived Wan2.2-Remix preset metadata marked as unsupported until sampler
  compatibility is implemented.

Not implemented yet:

- ComfyUI `KSamplerAdvanced` Euler/simple sampler compatibility.
- Executable `*-comfy-v1` presets.
- Pixel-level or node-level equivalence with ComfyUI reference workflows.

This document proposes a SGLang-native workflow execution service for diffusion
models. The service is intended for callers that are not ComfyUI but still need
more control than the current OpenAI-compatible `/v1/videos` endpoint exposes.

The motivating example is Wan2.2-Remix, whose reference usage is distributed as
ComfyUI workflow JSON files. The goal is not to execute ComfyUI graphs directly.
The goal is to translate the relevant generation semantics into a stable
SGLang workflow preset that can be requested by any client.

## Problem

The existing video generation API gives a simple product interface:

```text
prompt or image + prompt -> video
```

That is the right default for most users, but it hides pipeline details such as:

- sampler family and scheduler parameters
- timestep schedule and high/low-noise model split
- separate guidance scales for high-noise and low-noise experts
- text encoder variant and precision
- VAE variant and decode settings
- video encoding settings

For models like Wan2.2-Remix, those details are part of the reference workflow.
If SGLang always runs its built-in Wan pipeline defaults, it can provide a
functional video service, but it cannot claim node-level or pixel-level
equivalence with the reference workflow.

At the same time, exposing a low-level denoiser RPC such as:

```text
latent + timestep + prompt embeddings -> noise prediction
```

pushes too much diffusion-specific work onto non-Comfy callers. It also moves
large tensors across the service boundary for every denoising step unless the
API is made stateful and colocated with the caller.

## Goals

- Provide a generic workflow execution service that any client can call without
  depending on ComfyUI.
- Keep large tensors inside the SGLang runtime. The API boundary should carry
  prompts, images, preset names, scalar parameters, and output references, not
  per-step latents.
- Make workflow behavior explicit and reproducible through versioned presets.
- Support Wan2.2 dual-transformer workflows with separate high/low-noise
  experts.
- Allow per-request overrides for safe scalar knobs such as seed, resolution,
  number of frames, steps, guidance scales, and output options.
- Preserve the current simple `/v1/videos` path as the default product API.

## Non-Goals

- Do not implement a general ComfyUI graph executor inside SGLang.
- Do not expose a first-version HTTP endpoint that requires callers to upload
  latents or prompt embeddings per denoising step.
- Do not attempt to guarantee pixel-level equivalence with ComfyUI in the first
  version. The design should make the sources of drift explicit and reduce them
  incrementally.
- Do not require the caller to understand model component internals such as
  tensor names, latent packing, or VAE scale factors.

## High-Level Design

Add a workflow layer above existing SGLang diffusion pipelines:

```text
Client request
  -> workflow resolver
  -> workflow preset + request overrides
  -> workflow execution plan
  -> SGLang pipeline stages / model modules
  -> output storage
```

The workflow resolver selects a versioned preset. The preset describes the
generation semantics that are not obvious from the raw model path. The executor
then runs the complete generation process inside SGLang.

This is intentionally different from a denoiser service:

```text
Preferred:
  one request -> service-side workflow -> video

Avoid for first version:
  many requests -> caller-side scheduler loop -> per-step tensor RPC
```

## API Shape

### Endpoint

Add a SGLang-native endpoint:

```http
POST /v1/workflows/runs
```

The endpoint creates a workflow run and returns either an asynchronous run ID or
a completed output, matching the existing video API behavior.

### Request

```json
{
  "workflow": "wan2.2-remix/nsfw-t2v-comfy-v1",
  "input": {
    "prompt": "text prompt",
    "negative_prompt": "optional negative prompt",
    "image": null
  },
  "parameters": {
    "seed": 1024,
    "width": 1280,
    "height": 720,
    "num_frames": 81,
    "fps": 16,
    "num_inference_steps": 12,
    "guidance_scale": 1.0,
    "guidance_scale_2": 1.0
  },
  "output": {
    "format": "mp4",
    "response_format": "url"
  }
}
```

For image-to-video:

```json
{
  "workflow": "wan2.2-remix/nsfw-i2v-comfy-v1",
  "input": {
    "prompt": "text prompt",
    "negative_prompt": "optional negative prompt",
    "image_url": "https://example.com/input.png"
  },
  "parameters": {
    "seed": 1024,
    "width": 1280,
    "height": 720,
    "num_frames": 33,
    "fps": 16
  }
}
```

Multipart uploads should be supported for image inputs in the same style as the
current video API.

### Response

```json
{
  "id": "wf_run_...",
  "status": "completed",
  "workflow": "wan2.2-remix/nsfw-t2v-comfy-v1",
  "output": {
    "type": "video",
    "url": "/v1/workflows/runs/wf_run_.../content",
    "format": "mp4"
  },
  "effective_parameters": {
    "seed": 1024,
    "width": 1280,
    "height": 720,
    "num_frames": 81,
    "fps": 16,
    "num_inference_steps": 12,
    "guidance_scale": 1.0,
    "guidance_scale_2": 1.0,
    "sampler": "euler",
    "schedule": "simple",
    "flow_shift": 5.0
  }
}
```

## Workflow Preset Schema

Presets should be versioned data files checked into the SGLang tree or packaged
with model adapters. The first implementation can use JSON or YAML; JSON keeps
runtime dependencies minimal.

Example:

```json
{
  "schema_version": 1,
  "name": "wan2.2-remix/nsfw-t2v-comfy-v1",
  "task": "t2v",
  "model_family": "wan2.2",
  "pipeline": "WanWorkflowPipeline",
  "base_model_id": "Wan2.2-T2V-A14B-Diffusers",
  "components": {
    "transformer": "transformer",
    "transformer_2": "transformer_2",
    "text_encoder": "text_encoder",
    "tokenizer": "tokenizer",
    "vae": "vae"
  },
  "defaults": {
    "width": 1280,
    "height": 720,
    "num_frames": 81,
    "fps": 16,
    "seed": null,
    "num_inference_steps": 12,
    "guidance_scale": 1.0,
    "guidance_scale_2": 1.0
  },
  "sampler": {
    "type": "euler",
    "schedule": "simple",
    "flow_shift": 5.0
  },
  "experts": [
    {
      "name": "high_noise",
      "component": "transformer",
      "start_step": 0,
      "end_step": 10,
      "guidance_param": "guidance_scale"
    },
    {
      "name": "low_noise",
      "component": "transformer_2",
      "start_step": 10,
      "end_step": null,
      "guidance_param": "guidance_scale_2"
    }
  ],
  "decode": {
    "vae": "vae",
    "tiling": "auto"
  },
  "output": {
    "format": "mp4",
    "fps_param": "fps"
  },
  "override_policy": {
    "allow": [
      "prompt",
      "negative_prompt",
      "image",
      "seed",
      "width",
      "height",
      "num_frames",
      "fps",
      "num_inference_steps",
      "guidance_scale",
      "guidance_scale_2",
      "output.format",
      "output.response_format"
    ],
    "deny": [
      "components",
      "pipeline",
      "model_family"
    ]
  }
}
```

The preset is not a direct copy of a ComfyUI workflow. It records the execution
semantics that SGLang can own and reproduce.

## Execution Semantics

### Request Resolution

1. Parse the request and resolve `workflow` to a preset.
2. Validate that the currently served model can satisfy the preset:
   - task type matches (`t2v`, `i2v`, etc.)
   - required components exist
   - the model family matches, if specified
   - required optional capabilities exist, such as dual transformers
3. Merge request parameters over preset defaults according to
   `override_policy`.
4. Produce an immutable `WorkflowExecutionPlan`.

### Pipeline Integration

Add a workflow-aware pipeline path that can reuse existing modules:

- input validation
- text encoding
- optional image preprocessing and image conditioning
- latent initialization
- timestep preparation
- denoising loop
- VAE decoding
- output saving

For existing Wan models, this can be implemented by extending the current Wan
pipeline path instead of duplicating all modules.

The important new abstraction is the denoising controller. Existing SGLang Wan
uses `boundary_ratio` to decide when to switch from `transformer` to
`transformer_2`. A workflow preset needs to support explicit per-expert step
ranges as well:

```text
step i
  -> select expert by preset ranges
  -> select guidance scale by expert
  -> run model forward
  -> run sampler step
```

### Sampler Support

The first version should support two sampler modes:

1. Existing SGLang/Wan scheduler mode:
   - `FlowUniPCMultistepScheduler`
   - `boundary_ratio`
   - current high/low expert switch

2. Workflow sampler mode:
   - preset-selected sampler
   - explicit step ranges
   - preset-selected timestep/sigma schedule

For Wan2.2-Remix reference workflows, the missing piece is an Euler/simple
sampler compatible enough with the ComfyUI `KSamplerAdvanced` settings used by
the provided workflows. This should be added as a separate scheduler/sampler
implementation rather than hidden behind `boundary_ratio`.

### State and Tensor Lifetime

All large tensors should stay inside the workflow executor:

- prompt embeddings
- image conditioning embeddings
- latents
- per-step model outputs
- decoded frames

The public workflow API should not expose these tensors in the first version.
This avoids the performance and API stability problems of per-step tensor RPC.

If future clients need partial control without full tensor transfer, add
server-side handles:

```text
POST /v1/workflows/sessions
POST /v1/workflows/sessions/{id}/prepare
POST /v1/workflows/sessions/{id}/step
POST /v1/workflows/sessions/{id}/decode
```

Those handles should refer to tensors stored in the SGLang worker or shared
runtime, not serialized through HTTP.

## Wan2.2-Remix Mapping

The local Wan2.2-Remix model files are not complete Diffusers model trees. They
are Comfy/original-format transformer checkpoints. The conversion tool repacks
them into Diffusers/SGLang-compatible model directories and keeps the original
workflow JSON files only for provenance.

Recommended presets:

| Preset | Task | Model tree | Reference intent |
|--------|------|------------|------------------|
| `wan2.2-remix/sfw-t2v-sglang-v1` | T2V | `Wan2p2_T2V_A14B_Remix_SFW_v1.0_SGLang` | SGLang-native default generation |
| `wan2.2-remix/nsfw-t2v-sglang-v1` | T2V | `Wan2p2_T2V_A14B_Remix_NSFW_v2.0_SGLang` | SGLang-native default generation |
| `wan2.2-remix/nsfw-i2v-sglang-v1` | I2V | `Wan2p2_I2V_A14B_Remix_NSFW_v3.0_fp8_SGLang` | SGLang-native default generation |
| `wan2.2-remix/nsfw-t2v-comfy-v1` | T2V | selected converted T2V model | Approximate Comfy T2V workflow semantics |
| `wan2.2-remix/nsfw-i2v-comfy-v1` | I2V | selected converted I2V model | Approximate Comfy I2V workflow semantics |

The `*-sglang-v1` presets should use existing SGLang Wan behavior. The
`*-comfy-v1` presets encode the Comfy workflow-derived sampler settings and are
currently marked unsupported until SGLang can execute the required sampler.

Known Wan2.2-Remix reference workflow differences that a preset should record:

- T2V Comfy workflow:
  - resolution: `1280x720`
  - frames: `81`
  - fps: `16`
  - steps: `12`
  - sampler: `euler`
  - schedule: `simple`
  - high-noise step range: `0..10`
  - low-noise step range: `10..end`
  - CFG: `1.0` for both high and low workflow nodes
  - flow shift from `ModelSamplingSD3`: `5`
- I2V Comfy workflow:
  - resolution: `1280x720`
  - frames: `33`
  - fps: `16`
  - steps: `12`
  - sampler: `euler`
  - schedule: `simple`
  - high-noise step range: `0..10`
  - low-noise step range: `10..end`
  - CFG: `1.0` for both high and low workflow nodes
  - flow shift from `ModelSamplingSD3`: `8`
- Kijai T2V workflow:
  - resolution: `832x480`
  - frames: `81`
  - fps: `16`
  - sampler: `unipc`
  - high guidance: `3.5`
  - low guidance: `1.0`
  - high step range: `0..4`
  - low step range: `4..end`
  - flow shift: `8`

The original workflow files contain example prompts. Presets should not embed
those prompts unless the model owner explicitly wants a demo preset. Prompts
belong in requests.

## Code Changes

### Data Model

Add workflow dataclasses:

```text
python/sglang/multimodal_gen/configs/workflows/
  __init__.py
  schema.py
  registry.py
  presets/
    wan2_2_remix_sglang.json
    wan2_2_remix_comfy.json
```

Suggested classes:

- `WorkflowPreset`
- `WorkflowInput`
- `WorkflowParameters`
- `WorkflowOutputSpec`
- `WorkflowExecutionPlan`
- `WorkflowExpertRange`
- `WorkflowSamplerSpec`
- `WorkflowOverridePolicy`

### Registry

Add a workflow registry independent of the model registry:

```python
WorkflowRegistry.get(name: str) -> WorkflowPreset
WorkflowRegistry.list(model_family: str | None = None) -> list[WorkflowPreset]
WorkflowRegistry.validate(preset, server_args) -> None
```

The model registry identifies which pipeline and sampling parameter class a
model uses. The workflow registry identifies how to run a higher-level workflow
on top of an already served model.

### API Layer

Add endpoint definitions under the diffusion OpenAI runtime:

```text
python/sglang/multimodal_gen/runtime/entrypoints/openai/workflow_api.py
python/sglang/multimodal_gen/runtime/entrypoints/openai/workflow_protocol.py
```

Routes:

- `POST /v1/workflows/runs`
- `GET /v1/workflows/runs/{run_id}`
- `GET /v1/workflows/runs/{run_id}/content`
- `GET /v1/workflows`

The first version can reuse existing output storage and async video job
machinery where possible.

### Runtime

Add a workflow execution entrypoint:

```text
python/sglang/multimodal_gen/runtime/entrypoints/workflow_generator.py
```

Responsibilities:

- resolve preset
- merge defaults and overrides
- prepare `SamplingParams` or a workflow-specific request object
- dispatch to the scheduler
- return output metadata

For SGLang-native presets, this can internally call existing generation paths.
For Comfy-derived presets, it should dispatch to workflow-aware pipeline stages.

### Pipeline

Add workflow-aware denoising support without breaking existing pipelines:

- Keep current `DenoisingStage` behavior for normal requests.
- Add optional workflow denoising plan metadata under `batch.extra["workflow"]`.
- If `workflow_plan` is present:
  - use explicit expert ranges if present
  - use per-expert guidance scales
  - record effective parameters in output metrics

The current implementation preserves existing scheduler behavior. Workflow
sampler selection beyond `boundary_ratio` is deferred to Milestone 3.

Implementation candidates:

1. Add a small branch inside `DenoisingStage` for workflow plans.
2. Add a separate `WorkflowDenoisingStage` and wire it only into
   workflow-enabled pipelines.

The second option is cleaner if sampler semantics diverge significantly from
the current scheduler path.

### Wan Pipeline

For Wan2.2:

- Support `transformer` and `transformer_2` expert ranges.
- Support request-level override of `guidance_scale_2`; this already exists in
  sampling params.
- Add workflow sampler support for Euler/simple if Comfy-like presets are
  required.
- Preserve the current `boundary_ratio` path for normal `/v1/videos` requests.

## Validation Plan

### Unit Tests

- Parse valid and invalid workflow presets.
- Merge request overrides with defaults and enforce allow/deny policy.
- Validate task mismatch errors.
- Validate missing component errors.
- Validate expert ranges:
  - no negative starts
  - no overlapping ranges unless explicitly allowed
  - at least one range covers every requested denoising step
- Validate effective parameters returned to clients.

### Runtime Tests

- Run a tiny mocked workflow with a fake model and fake sampler.
- Run a Wan T2V workflow with `num_inference_steps=1` in CI-safe mode.
- Verify that normal `/v1/videos` behavior is unchanged.
- Verify that workflow presets do not require prompt text baked into presets.

### Model-Specific Tests

For Wan2.2-Remix on a real GPU environment:

- SGLang-native T2V preset produces an MP4.
- SGLang-native I2V preset produces an MP4 from an input image.
- Comfy-derived T2V preset accepts Comfy-equivalent scalar settings.
- Comfy-derived I2V preset accepts 33-frame workflow settings.
- Output metadata records the exact model path, preset version, sampler, step
  count, seed, resolution, frame count, and guidance scales.

## Compatibility and Migration

Existing APIs remain unchanged:

- `POST /v1/videos`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- CLI generation

Workflow execution is additive. The current video API may optionally accept a
simple `workflow` or `workflow_preset` parameter later, but the first
implementation should keep the new endpoint separate to avoid overloading the
OpenAI-compatible API.

## Open Questions

- Should workflow presets live only in the SGLang package, or can model
  directories provide `workflow_presets/*.json` that SGLang discovers at load
  time?
- Should Comfy-derived presets use exact Comfy sampler code, a compatible
  reimplementation, or SGLang's existing scheduler with approximation metadata?
- How much of video encoding should be part of the preset? CRF and pixel format
  affect binary output but not model sampling.
- Should workflow runs support partial artifacts such as latent trajectories for
  debugging?
- Should workflow sessions be added in the first version, or deferred until a
  real external workflow engine needs server-side handles?

## Recommended Milestones

### Milestone 1: Preset Metadata and SGLang-Native Workflow Runs

- Add workflow schema and registry.
- Add `POST /v1/workflows/runs`.
- Implement SGLang-native presets by translating into existing `SamplingParams`.
- Add Wan2.2-Remix `*-sglang-v1` presets.
- Return effective parameters and output content.

This milestone gives non-Comfy callers a stable workflow API without changing
the denoising loop.

Status: implemented.

### Milestone 2: Workflow-Aware Expert Selection

- Add `WorkflowExecutionPlan` to runtime requests.
- Support explicit high/low expert ranges for Wan2.2.
- Keep the current `boundary_ratio` behavior as the fallback.
- Add tests for expert selection.

This milestone makes dual-transformer workflows explicit.

Status: implemented for explicit expert ranges stored in workflow metadata.

### Milestone 3: Comfy-Derived Sampler Compatibility

- Implement Euler/simple sampler support needed by the Wan2.2-Remix Comfy
  workflows.
- Add `*-comfy-v1` presets.
- Compare effective timesteps and generated outputs against ComfyUI reference
  runs on the same weights and seed.

This milestone reduces workflow drift for users who care about Comfy-like
behavior.

### Milestone 4: Optional Stateful Workflow Sessions

- Add server-side tensor handles only if external workflow engines need
  step-level control.
- Keep tensors in SGLang worker memory or shared runtime state.
- Avoid serializing latents through HTTP.

This milestone is for advanced orchestration use cases and should not block the
generic workflow service.
