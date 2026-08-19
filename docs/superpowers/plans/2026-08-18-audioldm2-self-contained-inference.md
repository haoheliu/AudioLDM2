# AudioLDM2 Self-Contained Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-contained `audioldm2-full-large-1150k` prompt-to-WAV command with verbatim vendored model code, pinned Hugging Face artifacts, explicit codec identity, and a real end-to-end proof.

**Architecture:** `registry.py` is the typed source of truth for model geometry and artifact pins. `generator.py` constructs the vendored legacy model, loads its monolithic checkpoint strictly, and exposes one generation method; `run_inference.py` is the only public interface and shows the complete resolve → load → generate → validate → write workflow.

**Tech Stack:** Python 3.10+, PyTorch, torchaudio, Hugging Face Hub, Transformers 4.30.2, Click, SoundFile, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-audioldm2-self-contained-inference-design.md`

## Global Constraints

- Vendor every Git-tracked file below `audioldm2/` byte-for-byte from repository commit `b5786c5dc0ae8f766337fdc1b67ab6046586d14d`.
- Keep `self-contained/run_inference.py` as the only supported executable and public interface.
- Use only `haoheliu/audioldm2-full-large-1150k` revision `327bb3e09b48caeed6c283cfe16cdb2323dfb374`, file `audioldm2-full-large-1150k.pth`.
- Require checkpoint size `11,470,645,003` bytes and SHA-256 `6b483a47480d15d90c2eaaa17415484538bff7c6d9d2282f7838ad004c637c09`.
- Pin auxiliary assets to FLAN-T5 `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`, RoBERTa `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, and GPT-2 `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Fix output geometry at mono, 16 kHz, approximately 10 seconds; do not silently substitute another model, checkpoint, sample rate, or device.
- Refuse to overwrite output files and fail explicitly on artifact, device, state-dict, tensor, or WAV validation errors.
- Write each custom test before its production implementation and observe the expected failure.
- The final acceptance run must execute real conditioning, DDIM, spectrogram VAE decode, and HiFi-GAN vocoding; mocks are forbidden in the acceptance run.

---

## File Structure

- `self-contained/run_inference.py`: Click command, `RunConfig`, and the read-first orchestration function.
- `self-contained/registry.py`: immutable model/artifact specifications, hashing, checkpoint resolution, and auxiliary asset prefetch.
- `self-contained/generator.py`: vendored API loading, strict model lifecycle, prompt batching, and waveform generation.
- `self-contained/requirements.txt`: complete runtime dependencies for the legacy source path.
- `self-contained/vendor/LICENSE`: verbatim upstream repository license.
- `self-contained/vendor/audioldm2/**`: verbatim Git-tracked upstream package and assets.
- `tests/test_self_contained_vendor.py`: vendor membership and byte-parity contract.
- `tests/test_self_contained_registry.py`: artifact identity, corruption rejection, and auxiliary layout tests.
- `tests/test_self_contained_generator.py`: large-config selection, strict loading, and generation orchestration tests.
- `tests/test_self_contained_cli.py`: CLI validation, no-overwrite behavior, and WAV contract tests.
- `tests/test_self_contained_e2e.py`: opt-in real checkpoint-to-WAV acceptance check.

### Task 1: Verbatim Vendor Boundary

**Files:**
- Create: `self-contained/vendor/LICENSE`
- Create: `self-contained/vendor/audioldm2/**`
- Test: `tests/test_self_contained_vendor.py`

**Interfaces:**
- Consumes: Git-tracked source files from `audioldm2/` and root `LICENSE`.
- Produces: `self-contained/vendor/audioldm2` import root with byte-identical source and assets.

- [ ] **Step 1: Write the failing vendor parity test**

```python
from hashlib import sha256
from pathlib import Path
import subprocess


root = Path(__file__).resolve().parents[1]
source_root = root / "audioldm2"
vendor_root = root / "self-contained" / "vendor" / "audioldm2"


def tracked_package_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "audioldm2"], cwd=root, text=True
    )
    return [root / line for line in output.splitlines() if line]


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_vendor_tree_matches_every_tracked_package_file() -> None:
    tracked = tracked_package_files()
    expected = {path.relative_to(source_root) for path in tracked}
    actual = {
        path.relative_to(vendor_root)
        for path in vendor_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert actual == expected
    for relative_path in sorted(expected):
        assert file_digest(vendor_root / relative_path) == file_digest(source_root / relative_path)


def test_vendor_license_is_verbatim() -> None:
    assert file_digest(root / "self-contained" / "vendor" / "LICENSE") == file_digest(
        root / "LICENSE"
    )
```

- [ ] **Step 2: Run the test and verify the missing vendor tree fails**

Run: `python -m pytest tests/test_self_contained_vendor.py -v`

Expected: FAIL because `self-contained/vendor/audioldm2` and its license do not exist.

- [ ] **Step 3: Copy only tracked upstream files without modifying content**

Run:

```bash
mkdir -p self-contained/vendor
git ls-files -z audioldm2 | xargs -0 cp --parents --target-directory=self-contained/vendor
cp LICENSE self-contained/vendor/LICENSE
```

- [ ] **Step 4: Re-run the vendor parity test**

Run: `python -m pytest tests/test_self_contained_vendor.py -v`

Expected: 2 passed; every tracked relative path and digest matches.

- [ ] **Step 5: Commit the vendor boundary**

```bash
git add self-contained/vendor tests/test_self_contained_vendor.py
git commit -m "vendor: preserve AudioLDM2 inference source"
```

### Task 2: Typed Registry and Pinned Artifact Resolver

**Files:**
- Create: `self-contained/registry.py`
- Test: `tests/test_self_contained_registry.py`

**Interfaces:**
- Consumes: `huggingface_hub.hf_hub_download`, `huggingface_hub.snapshot_download`, and caller-owned cache roots.
- Produces: `ModelSpec`, `ArtifactSpec`, `model_spec()`, `sha256_file(path)`, `resolve_checkpoint(spec, cache_dir)`, and `prefetch_auxiliary_assets(spec, asset_root)`.

- [ ] **Step 1: Write failing registry identity and corruption tests**

```python
from pathlib import Path
import sys

import pytest


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "self-contained"))
import registry


def test_model_spec_pins_large_checkpoint_and_native_audio() -> None:
    spec = registry.model_spec()
    assert spec.name == "audioldm2-full-large-1150k"
    assert spec.sample_rate == 16000
    assert spec.duration_seconds == 10.0
    assert spec.checkpoint.repo_id == "haoheliu/audioldm2-full-large-1150k"
    assert spec.checkpoint.revision == "327bb3e09b48caeed6c283cfe16cdb2323dfb374"
    assert spec.checkpoint.size_bytes == 11470645003
    assert spec.checkpoint.sha256 == "6b483a47480d15d90c2eaaa17415484538bff7c6d9d2282f7838ad004c637c09"


def test_validate_artifact_rejects_wrong_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pth"
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        registry.validate_artifact(artifact, registry.model_spec().checkpoint)


def test_auxiliary_assets_have_local_names_used_by_vendor() -> None:
    assets = {item.local_name: item for item in registry.model_spec().auxiliary_assets}
    assert set(assets) == {"google/flan-t5-large", "roberta-base", "gpt2"}
    assert assets["google/flan-t5-large"].revision == "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    assert assets["roberta-base"].revision == "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
    assert assets["gpt2"].revision == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
```

- [ ] **Step 2: Verify the registry import fails**

Run: `python -m pytest tests/test_self_contained_registry.py -v`

Expected: collection ERROR with `ModuleNotFoundError: No module named 'registry'`.

- [ ] **Step 3: Implement immutable specifications and artifact validation**

```python
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


@dataclass(frozen=True)
class ArtifactSpec:
    repo_id: str
    revision: str
    filename: str = ""
    local_name: str = ""
    size_bytes: int = 0
    sha256: str = ""
    allow_patterns: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    sample_rate: int
    duration_seconds: float
    latent_t_per_second: float
    default_ddim_steps: int
    default_guidance_scale: float
    default_candidates: int
    checkpoint: ArtifactSpec
    auxiliary_assets: tuple[ArtifactSpec, ...]


def model_spec() -> ModelSpec:
    checkpoint = ArtifactSpec(
        repo_id="haoheliu/audioldm2-full-large-1150k",
        revision="327bb3e09b48caeed6c283cfe16cdb2323dfb374",
        filename="audioldm2-full-large-1150k.pth",
        size_bytes=11470645003,
        sha256="6b483a47480d15d90c2eaaa17415484538bff7c6d9d2282f7838ad004c637c09",
    )
    auxiliary_assets = (
        ArtifactSpec(
            repo_id="google/flan-t5-large",
            revision="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
            local_name="google/flan-t5-large",
            allow_patterns=("config.json", "spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
        ),
        ArtifactSpec(
            repo_id="FacebookAI/roberta-base",
            revision="e2da8e2f811d1448a5b465c236feacd80ffbac7b",
            local_name="roberta-base",
            allow_patterns=("config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"),
        ),
        ArtifactSpec(
            repo_id="openai-community/gpt2",
            revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
            local_name="gpt2",
            allow_patterns=("config.json",),
        ),
    )
    return ModelSpec(
        name="audioldm2-full-large-1150k",
        sample_rate=16000,
        duration_seconds=10.0,
        latent_t_per_second=25.6,
        default_ddim_steps=200,
        default_guidance_scale=3.5,
        default_candidates=3,
        checkpoint=checkpoint,
        auxiliary_assets=auxiliary_assets,
    )


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(path: Path, artifact: ArtifactSpec) -> None:
    if path.stat().st_size != artifact.size_bytes:
        raise ValueError(f"checkpoint size mismatch: {path}")
    if sha256_file(path) != artifact.sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")


def resolve_checkpoint(spec: ModelSpec, cache_dir: Path) -> Path:
    path = Path(hf_hub_download(
        repo_id=spec.checkpoint.repo_id,
        filename=spec.checkpoint.filename,
        revision=spec.checkpoint.revision,
        cache_dir=cache_dir,
    ))
    validate_artifact(path, spec.checkpoint)
    return path


def prefetch_auxiliary_assets(spec: ModelSpec, asset_root: Path) -> None:
    for artifact in spec.auxiliary_assets:
        snapshot_download(
            repo_id=artifact.repo_id,
            revision=artifact.revision,
            allow_patterns=list(artifact.allow_patterns),
            local_dir=asset_root / artifact.local_name,
        )
```

- [ ] **Step 4: Run registry tests**

Run: `python -m pytest tests/test_self_contained_registry.py -v`

Expected: 3 passed without downloading the 11.47 GB checkpoint.

- [ ] **Step 5: Commit the registry**

```bash
git add self-contained/registry.py tests/test_self_contained_registry.py
git commit -m "feat: pin AudioLDM2 inference artifacts"
```

### Task 3: Strict Generator Wrapper

**Files:**
- Create: `self-contained/generator.py`
- Test: `tests/test_self_contained_generator.py`

**Interfaces:**
- Consumes: `registry.ModelSpec`, a verified checkpoint path, auxiliary asset root, device string, and optional injected `VendorApi` for fast tests.
- Produces: `VendorApi`, `load_vendor_api(vendor_root)`, and `AudioLDM2Generator.generate(prompt, seed, ddim_steps, guidance_scale, candidates) -> numpy.ndarray` with shape `[batch, channel, samples]`.

- [ ] **Step 1: Write failing construction and generation tests with a fake vendored boundary**

```python
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "self-contained"))
import generator
import registry


class FakeLatentDiffusion(torch.nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.params = params
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.latent_t_size = 0
        self.loaded_strict = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def generate_batch(self, batch, unconditional_guidance_scale, ddim_steps, n_gen, duration):
        assert batch["text"] == ["rain on a window"]
        assert unconditional_guidance_scale == 3.5
        assert ddim_steps == 2
        assert n_gen == 1
        assert duration == 10.0
        return np.zeros((1, 1, 160000), dtype=np.float32)


def fake_config(name):
    assert name == "audioldm2-full-large-1150k"
    return {
        "model": {
            "params": {
                "device": "cpu",
                "unet_config": {"params": {"context_dim": [768, 1024, None], "transformer_depth": 2}},
            }
        }
    }


def fake_batch(prompt, transcription, batchsize):
    return {"text": [prompt] * batchsize, "transcription": transcription}


def test_generator_selects_large_config_and_loads_strictly(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    api = generator.VendorApi(FakeLatentDiffusion, fake_config, fake_batch)
    model = generator.AudioLDM2Generator(
        registry.model_spec(), checkpoint, tmp_path, "cpu", api=api
    )
    assert model.model.params["unet_config"]["params"]["transformer_depth"] == 2
    assert model.model.loaded_strict is True


def test_generate_runs_vendored_batch_and_model_path(tmp_path) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    api = generator.VendorApi(FakeLatentDiffusion, fake_config, fake_batch)
    model = generator.AudioLDM2Generator(
        registry.model_spec(), checkpoint, tmp_path, "cpu", api=api
    )
    waveform = model.generate("rain on a window", seed=7, ddim_steps=2, guidance_scale=3.5, candidates=1)
    assert waveform.shape == (1, 1, 160000)
    assert np.isfinite(waveform).all()
```

- [ ] **Step 2: Verify generator import fails**

Run: `python -m pytest tests/test_self_contained_generator.py -v`

Expected: collection ERROR with `ModuleNotFoundError: No module named 'generator'`.

- [ ] **Step 3: Implement vendored API loading and strict model lifecycle**

```python
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import random
import sys
from typing import Callable

import numpy as np
import torch

from registry import ModelSpec


@dataclass(frozen=True)
class VendorApi:
    latent_diffusion_type: type
    default_config: Callable
    make_batch: Callable


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_vendor_api(vendor_root: Path) -> VendorApi:
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    from audioldm2.latent_diffusion.models.ddpm import LatentDiffusion
    from audioldm2.pipeline import make_batch_for_text_to_audio
    from audioldm2.utils import default_audioldm_config
    return VendorApi(LatentDiffusion, default_audioldm_config, make_batch_for_text_to_audio)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AudioLDM2Generator:
    def __init__(self, spec: ModelSpec, checkpoint_path: Path, asset_root: Path,
                 device: str, api: VendorApi | None = None):
        self.spec = spec
        self.device = torch.device(device)
        vendor_root = Path(__file__).resolve().parent / "vendor"
        self.api = api or load_vendor_api(vendor_root)
        with working_directory(asset_root):
            config = self.api.default_config(spec.name)
            config["model"]["params"]["device"] = self.device
            self.model = self.api.latent_diffusion_type(**config["model"]["params"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        gc.collect()
        self.model.eval().to(self.device)

    @torch.no_grad()
    def generate(self, prompt: str, seed: int, ddim_steps: int,
                 guidance_scale: float, candidates: int) -> np.ndarray:
        seed_everything(seed)
        batch = self.api.make_batch(prompt, transcription="", batchsize=1)
        self.model.latent_t_size = int(self.spec.duration_seconds * self.spec.latent_t_per_second)
        waveform = self.model.generate_batch(
            batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_gen=candidates,
            duration=self.spec.duration_seconds,
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 3 or waveform.shape[1] != 1 or not np.isfinite(waveform).all():
            raise ValueError(f"invalid waveform tensor: shape={waveform.shape}")
        return waveform
```

- [ ] **Step 4: Run generator tests**

Run: `python -m pytest tests/test_self_contained_generator.py -v`

Expected: 2 passed, including strict state loading and the full wrapper call sequence.

- [ ] **Step 5: Commit the generator wrapper**

```bash
git add self-contained/generator.py tests/test_self_contained_generator.py
git commit -m "feat: wrap strict AudioLDM2 generation"
```

### Task 4: Single Public Click Runner

**Files:**
- Create: `self-contained/run_inference.py`
- Create: `self-contained/requirements.txt`
- Test: `tests/test_self_contained_cli.py`

**Interfaces:**
- Consumes: `registry.model_spec`, artifact resolution functions, `AudioLDM2Generator`, and CLI values.
- Produces: `RunConfig`, `choose_device(requested)`, `validate_waveform(waveform, spec)`, `run_inference(config) -> Path`, and Click command `main`.

- [ ] **Step 1: Write failing CLI validation tests**

```python
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest
from click.testing import CliRunner


root = Path(__file__).resolve().parents[1]
self_contained = root / "self-contained"
sys.path.insert(0, str(self_contained))
module_spec = spec_from_file_location("run_inference", self_contained / "run_inference.py")
run_inference = module_from_spec(module_spec)
module_spec.loader.exec_module(run_inference)


def test_cli_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "exists.wav"
    output.write_bytes(b"keep")
    result = CliRunner().invoke(
        run_inference.main,
        ["--prompt", "rain", "--output", str(output), "--device", "cpu"],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_bytes() == b"keep"


def test_validate_waveform_rejects_nonfinite_audio() -> None:
    waveform = np.array([[[np.nan, 0.0]]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        run_inference.validate_waveform(waveform, run_inference.model_spec())


def test_choose_device_rejects_unavailable_cuda(monkeypatch) -> None:
    monkeypatch.setattr(run_inference.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA was requested"):
        run_inference.choose_device("cuda")
```

- [ ] **Step 2: Verify runner import fails**

Run: `python -m pytest tests/test_self_contained_cli.py -v`

Expected: collection ERROR because `self-contained/run_inference.py` does not exist.

- [ ] **Step 3: Implement the read-first workflow and Click command**

```python
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import soundfile as sf
import torch

from generator import AudioLDM2Generator
from registry import ModelSpec, model_spec, prefetch_auxiliary_assets, resolve_checkpoint


@dataclass(frozen=True)
class RunConfig:
    prompt: str
    output: Path
    cache_dir: Path
    device: str
    seed: int
    ddim_steps: int
    guidance_scale: float
    candidates: int


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def validate_waveform(waveform: np.ndarray, spec: ModelSpec) -> None:
    if waveform.ndim != 3 or waveform.shape[0] < 1 or waveform.shape[1] != 1:
        raise ValueError(f"expected [candidate, 1, samples], got {waveform.shape}")
    if waveform.shape[-1] < int(spec.sample_rate * 9.5):
        raise ValueError(f"waveform is too short: {waveform.shape[-1]} samples")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains non-finite samples")


def run_inference(config: RunConfig) -> Path:
    if config.output.exists():
        raise FileExistsError(f"output already exists: {config.output}")
    spec = model_spec()
    device = choose_device(config.device)
    asset_root = config.cache_dir / "assets"
    prefetch_auxiliary_assets(spec, asset_root)
    checkpoint = resolve_checkpoint(spec, config.cache_dir / "huggingface")
    generator = AudioLDM2Generator(spec, checkpoint, asset_root, device)
    waveform = generator.generate(
        config.prompt, config.seed, config.ddim_steps,
        config.guidance_scale, config.candidates,
    )
    validate_waveform(waveform, spec)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(config.output, waveform[0, 0], spec.sample_rate, subtype="PCM_16")
    return config.output


@click.command()
@click.option("--prompt", required=True, type=str)
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--cache-dir", type=click.Path(path_type=Path), default=Path(".cache/audioldm2"))
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto")
@click.option("--seed", type=int, default=0)
@click.option("--ddim-steps", type=click.IntRange(min=1), default=200)
@click.option("--guidance-scale", type=click.FloatRange(min=0.0), default=3.5)
@click.option("--candidates", type=click.IntRange(min=1), default=3)
def main(prompt: str, output: Path, cache_dir: Path, device: str, seed: int,
         ddim_steps: int, guidance_scale: float, candidates: int) -> None:
    config = RunConfig(prompt, output, cache_dir, device, seed, ddim_steps, guidance_scale, candidates)
    try:
        result = run_inference(config)
    except (FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote {result}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the explicit legacy runtime requirements**

```text
click>=8.1,<9
einops
ftfy
huggingface_hub>=0.16
librosa==0.9.2
matplotlib
numpy<=1.23.5
pandas
phonemizer
Pillow
progressbar2
pyyaml
regex
scikit-learn
scipy
soundfile
timm
torch>=2.0
torchaudio
torchlibrosa>=0.0.9
torchvision
transformers==4.30.2
tqdm
unidecode
```

- [ ] **Step 5: Run CLI and all fast tests**

Run:

```bash
python -m pytest \
  tests/test_self_contained_vendor.py \
  tests/test_self_contained_registry.py \
  tests/test_self_contained_generator.py \
  tests/test_self_contained_cli.py -v
python self-contained/run_inference.py --help
```

Expected: all tests pass; help shows one command with prompt/output/cache/device/sampling options.

- [ ] **Step 6: Commit the single public runner**

```bash
git add self-contained/run_inference.py self-contained/requirements.txt tests/test_self_contained_cli.py
git commit -m "feat: expose AudioLDM2 inference command"
```

### Task 5: Real Artifact Acquisition and End-to-End Proof

**Files:**
- Create: `tests/test_self_contained_e2e.py`
- Generated and ignored: `.cache/audioldm2/**`
- Generated and ignored: `self-contained/output/e2e-proof.wav`

**Interfaces:**
- Consumes: the exact public CLI from Task 4 and its pinned resolver.
- Produces: a real mono 16 kHz WAV and machine-checked proof that every inference component executed successfully.

- [ ] **Step 1: Write the opt-in acceptance test before running inference**

```python
from pathlib import Path
import os
import subprocess

import numpy as np
import pytest
import soundfile as sf


root = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("AUDIOLDM2_E2E") != "1",
    reason="set AUDIOLDM2_E2E=1 to run the 11.47 GB checkpoint acceptance test",
)
def test_real_large_checkpoint_generates_valid_waveform(tmp_path: Path) -> None:
    output = tmp_path / "e2e.wav"
    subprocess.run(
        [
            "python", "self-contained/run_inference.py",
            "--prompt", "A gentle rainstorm outside a quiet cabin",
            "--output", str(output),
            "--cache-dir", str(root / ".cache" / "audioldm2"),
            "--device", "cuda",
            "--seed", "0",
            "--ddim-steps", "2",
            "--guidance-scale", "3.5",
            "--candidates", "1",
        ],
        cwd=root,
        check=True,
    )
    audio, sample_rate = sf.read(output, dtype="float32", always_2d=True)
    assert sample_rate == 16000
    assert audio.shape[1] == 1
    assert 152000 <= audio.shape[0] <= 164000
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 1e-5
```

- [ ] **Step 2: Verify the acceptance test is skipped by default**

Run: `python -m pytest tests/test_self_contained_e2e.py -v`

Expected: 1 skipped with the explicit `AUDIOLDM2_E2E` instruction.

- [ ] **Step 3: Gather and verify every pinned artifact**

Run:

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'self-contained'); from registry import model_spec, prefetch_auxiliary_assets, resolve_checkpoint; s=model_spec(); r=Path('.cache/audioldm2'); prefetch_auxiliary_assets(s, r/'assets'); print(resolve_checkpoint(s, r/'huggingface'))"
```

Expected: the three auxiliary asset directories exist and the downloaded `.pth` passes both the exact 11,470,645,003-byte check and SHA-256 check.

- [ ] **Step 4: Run the real checkpoint-to-WAV acceptance test**

Run: `AUDIOLDM2_E2E=1 python -m pytest tests/test_self_contained_e2e.py -v -s`

Expected: 1 passed after strict model loading, two DDIM steps, VAE mel decode, and HiFi-GAN waveform synthesis.

- [ ] **Step 5: Run a persistent public-command proof artifact**

Run:

```bash
python self-contained/run_inference.py \
  --prompt "A gentle rainstorm outside a quiet cabin" \
  --output self-contained/output/e2e-proof.wav \
  --cache-dir .cache/audioldm2 \
  --device cuda \
  --seed 0 \
  --ddim-steps 2 \
  --guidance-scale 3.5 \
  --candidates 1
```

Expected: command exits 0 and writes a finite, non-silent mono 16 kHz WAV of approximately ten seconds.

- [ ] **Step 6: Re-run the complete fast and acceptance suites**

Run:

```bash
python -m pytest \
  tests/test_self_contained_vendor.py \
  tests/test_self_contained_registry.py \
  tests/test_self_contained_generator.py \
  tests/test_self_contained_cli.py -v
AUDIOLDM2_E2E=1 python -m pytest tests/test_self_contained_e2e.py -v -s
```

Expected: all fast tests pass and the real acceptance test passes with no mocks.

- [ ] **Step 7: Commit the acceptance contract**

```bash
git add tests/test_self_contained_e2e.py
git commit -m "test: prove AudioLDM2 end-to-end inference"
```

### Task 6: Final Requirement and Provenance Audit

**Files:**
- Modify only if a discovered mismatch requires correction: `self-contained/registry.py`, `self-contained/generator.py`, `self-contained/run_inference.py`, `self-contained/requirements.txt`, or their tests.

**Interfaces:**
- Consumes: design spec, implementation diff, test results, checkpoint hash, and generated WAV metadata.
- Produces: evidence-backed completion report with no unverified success claims.

- [ ] **Step 1: Verify vendor parity after all imports and tests**

Run: `python -m pytest tests/test_self_contained_vendor.py -v`

Expected: 2 passed even if Python created ignored `__pycache__` files.

- [ ] **Step 2: Inspect the exact implementation diff and public surface**

Run:

```bash
git diff --check
git status --short
find self-contained -maxdepth 2 -type f -not -path '*/__pycache__/*' | sort
python self-contained/run_inference.py --help
```

Expected: no whitespace errors; only `run_inference.py` is executable/public custom code; internal registry/generator and vendored source match the approved map.

- [ ] **Step 3: Verify checkpoint and WAV evidence directly**

Run:

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'self-contained'); from registry import model_spec, resolve_checkpoint; s=model_spec(); print(resolve_checkpoint(s, Path('.cache/audioldm2/huggingface')))"
python -c "import soundfile as sf; from pathlib import Path; p=Path('self-contained/output/e2e-proof.wav'); x,sr=sf.read(p, dtype='float32', always_2d=True); print({'path':str(p),'sample_rate':sr,'frames':len(x),'channels':x.shape[1],'peak':float(abs(x).max())})"
```

Expected: checkpoint validation succeeds; WAV reports 16,000 Hz, one channel, roughly 160,000 frames, and nonzero finite peak.

- [ ] **Step 4: Run the full fresh verification command**

Run:

```bash
python -m pytest \
  tests/test_self_contained_vendor.py \
  tests/test_self_contained_registry.py \
  tests/test_self_contained_generator.py \
  tests/test_self_contained_cli.py -v
AUDIOLDM2_E2E=1 python -m pytest tests/test_self_contained_e2e.py -v -s
```

Expected: zero failures; any skipped or failed test blocks completion.

- [ ] **Step 5: Confirm the final commit set**

Run: `git log --oneline --decorate -8`

Expected: separate commits for vendor source, artifact registry, generator wrapper, public runner, and end-to-end contract.
