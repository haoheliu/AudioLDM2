# Self-Contained AudioLDM2 Inference Design

Date: 2026-08-18

## Goal

Create a source-contained AudioLDM2 inference package under `self-contained/` for the `audioldm2-full-large-1150k` model. The package must preserve upstream model code verbatim, expose one simple public `run_inference.py` command, acquire every required Hugging Face artifact reproducibly, identify the audio codec path, and prove real prompt-to-WAV inference.

## Scope

The implementation supports text-to-audio generation only. It does not expose training, text-to-speech transcription controls, audio-guided generation, super-resolution, inpainting, Gradio, or batch-list interfaces.

The public command accepts a prompt, output path, device, seed, DDIM step count, guidance scale, and candidate count. AudioLDM2 Large is fixed to its native 10-second, 16 kHz output protocol.

## Source and Artifact Identity

The vendored `audioldm2/` package is copied from this repository at commit `b5786c5dc0ae8f766337fdc1b67ab6046586d14d`. Every copied file must match its source byte-for-byte. A verification test compares the source and vendor trees by relative path and SHA-256 digest.

The primary checkpoint is:

- Repository: `haoheliu/audioldm2-full-large-1150k`
- Revision: `327bb3e09b48caeed6c283cfe16cdb2323dfb374`
- File: `audioldm2-full-large-1150k.pth`
- Size: 11,470,645,003 bytes
- SHA-256: `6b483a47480d15d90c2eaaa17415484538bff7c6d9d2282f7838ad004c637c09`

The checkpoint contains the strict `LatentDiffusion.state_dict`, including the large UNet, conditioning models, spectrogram autoencoder, and HiFi-GAN weights. Construction also needs tokenizer or configuration assets, but not additional model checkpoints:

- `google/flan-t5-large` at `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`
- `FacebookAI/roberta-base` at `e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- `openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`

Artifact acquisition fails explicitly if a pinned file cannot be downloaded or if the main checkpoint's size or SHA-256 differs. Checkpoints and Hugging Face cache data remain outside Git.

## Architecture

The package follows the conventions of `src/models/audioX`:

- `registry.py` contains frozen dataclass specifications for model geometry, sampling defaults, source pins, and artifact identity.
- `generator.py` owns model construction, strict checkpoint loading, device placement, batch construction, and generation.
- `vendor/audioldm2/` contains only verbatim upstream source and package assets.
- `run_inference.py` is the sole public interface and visibly shows the full workflow.

No custom model layer reimplements upstream math. The wrapper calls the vendored configuration, `LatentDiffusion`, batching, and generation functions directly. It does not call upstream `pipeline.build_model`, because that function ignores its `ckpt_path` argument and downloads an unpinned checkpoint.

## Inference Data Flow

1. `run_inference.py` validates arguments and refuses to overwrite an existing WAV.
2. The artifact resolver downloads or reuses the pinned checkpoint and auxiliary tokenizer/config snapshots.
3. `AudioLDM2Generator` creates the large configuration. The `-large-` model rule selects UNet context dimensions `[768, 1024, None]` and transformer depth 2.
4. `LatentDiffusion` constructs the conditioning stack, diffusion model, and first stage.
5. The checkpoint `state_dict` loads strictly before the model enters evaluation mode on the selected device.
6. Prompt batching creates text, placeholder waveform/STFT/log-mel tensors, and phoneme IDs.
7. CLAP and FLAN-T5 encode the prompt. A GPT-2 sequence model produces AudioMAE-space conditioning tokens; the phoneme encoder supports the shared batch contract.
8. DDIM performs classifier-free reverse diffusion in the eight-channel latent spectrogram space.
9. The first-stage autoencoder decodes the latent into a 64-bin log-mel spectrogram.
10. HiFi-GAN converts the mel spectrogram into a mono 16 kHz waveform, which is written to the requested WAV.

## Audio Codec Identification

AudioLDM2 does not use a waveform tokenizer such as EnCodec. Its first stage is a KL-regularized convolutional variational autoencoder over log-mel spectrograms:

- Encoder output is projected by `quant_conv` into a diagonal Gaussian posterior.
- An eight-channel latent sample is denoised by the diffusion model.
- `post_quant_conv` and the convolutional decoder reconstruct a 64-bin mel spectrogram.
- A separately instantiated HiFi-GAN generator performs mel-to-waveform synthesis at 16 kHz.

Both autoencoder and vocoder parameters are restored from the single main checkpoint through strict model loading.

## File Map

```text
self-contained/
|-- run_inference.py
|-- registry.py
|-- generator.py
|-- requirements.txt
`-- vendor/
    |-- LICENSE
    `-- audioldm2/

tests/
`-- test_self_contained_inference.py
```

`run_inference.py` is the only supported executable or import surface. `registry.py` and `generator.py` are internal implementation modules and do not define additional commands.

## Failure Behavior

The command fails before model construction when the output exists, CUDA was requested but is unavailable, an artifact is missing or corrupt, or dependencies are incompatible. Checkpoint loading is strict. Unexpected generation shapes, sample rates, empty audio, non-finite samples, and invalid WAV metadata fail the end-to-end proof.

There is no silent CPU fallback when a specific device is requested and no fallback to another checkpoint or Diffusers model.

## Verification

Tests are written before the custom implementation.

Fast tests cover:

- the fixed model specification and artifact pins;
- output overwrite rejection and CLI validation;
- exact vendor-tree file membership and SHA-256 parity;
- checkpoint resolver size and digest validation;
- model construction selecting the large UNet configuration;
- inference orchestration through real vendored functions with only heavyweight tensor execution replaced at the boundary.

The acceptance run is not mocked. It must:

1. download or locate the pinned 11.47 GB checkpoint and auxiliary assets;
2. verify the checkpoint SHA-256;
3. instantiate the complete model and load every state-dict key strictly;
4. generate audio from a fixed text prompt;
5. execute the VAE decoder and HiFi-GAN vocoder;
6. write a WAV and verify 16 kHz mono metadata, finite non-empty samples, and approximately ten seconds of audio.

The generated proof WAV is an ignored test artifact, not a repository fixture.

## Constraints and Risks

The 1.5B-parameter large model and 11.47 GB checkpoint require substantial RAM, GPU memory, download time, and inference time. Full 200-step, three-candidate generation is the upstream quality protocol; the end-to-end proof may use fewer DDIM steps and one candidate to bound validation cost while still exercising every model component. The public defaults remain aligned with upstream inference practice.

The checkpoint's Hugging Face repository is distinct from the Diffusers-format alias `cvssp/audioldm2-large`. This implementation uses the legacy `.pth` artifact because the vendored source expects the original monolithic state dictionary.
