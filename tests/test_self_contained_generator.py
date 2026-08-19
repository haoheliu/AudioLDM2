import importlib
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
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

    def generate_batch(
        self,
        batch,
        unconditional_guidance_scale,
        ddim_steps,
        n_gen,
        duration,
    ):
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
                "unet_config": {
                    "params": {
                        "context_dim": [768, 1024, None],
                        "transformer_depth": 2,
                    }
                },
            }
        }
    }


def fake_batch(prompt, transcription, batchsize):
    return {"text": [prompt] * batchsize, "transcription": transcription}


def lightweight_real_vendor_api(monkeypatch) -> generator.VendorApi:
    vendor_root = root / "self-contained" / "vendor" / "audioldm2"
    package = ModuleType("audioldm2")
    package.__path__ = [str(vendor_root)]
    monkeypatch.setitem(sys.modules, "audioldm2", package)

    ddpm = ModuleType("audioldm2.latent_diffusion.models.ddpm")
    ddpm.LatentDiffusion = FakeLatentDiffusion
    monkeypatch.setitem(
        sys.modules,
        "audioldm2.latent_diffusion.models.ddpm",
        ddpm,
    )
    pipeline = importlib.import_module("audioldm2.pipeline")
    utils = importlib.import_module("audioldm2.utils")
    return generator.VendorApi(
        FakeLatentDiffusion,
        utils.default_audioldm_config,
        pipeline.make_batch_for_text_to_audio,
    )


def test_vendor_import_runs_from_pinned_asset_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    api = generator.VendorApi(FakeLatentDiffusion, fake_config, fake_batch)

    def fake_loader(vendor_root: Path):
        assert Path.cwd() == tmp_path
        return api

    monkeypatch.setattr(generator, "load_vendor_api", fake_loader)
    generator.AudioLDM2Generator(
        registry.model_spec(), checkpoint, tmp_path, "cpu"
    )


def test_generator_uses_real_vendored_config_and_batch_functions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    api = lightweight_real_vendor_api(monkeypatch)
    model = generator.AudioLDM2Generator(
        registry.model_spec(), checkpoint, tmp_path, "cpu", api=api
    )
    unet = model.model.params["unet_config"]["params"]
    assert unet["context_dim"] == [768, 1024, None]
    assert unet["transformer_depth"] == 2
    waveform = model.generate(
        "rain on a window",
        seed=7,
        ddim_steps=2,
        guidance_scale=3.5,
        candidates=1,
    )
    assert waveform.shape == (1, 1, 160000)


def test_generator_selects_large_config_and_loads_strictly(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    api = generator.VendorApi(FakeLatentDiffusion, fake_config, fake_batch)
    model = generator.AudioLDM2Generator(
        registry.model_spec(), checkpoint, tmp_path, "cpu", api=api
    )
    unet = model.model.params["unet_config"]["params"]
    assert unet["context_dim"] == [768, 1024, None]
    assert unet["transformer_depth"] == 2
    assert model.model.loaded_strict is True


def test_generate_runs_vendored_batch_and_model_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
    api = generator.VendorApi(FakeLatentDiffusion, fake_config, fake_batch)
    model = generator.AudioLDM2Generator(
        registry.model_spec(), checkpoint, tmp_path, "cpu", api=api
    )
    waveform = model.generate(
        "rain on a window",
        seed=7,
        ddim_steps=2,
        guidance_scale=3.5,
        candidates=1,
    )
    assert model.model.latent_t_size == 256
    assert waveform.shape == (1, 1, 160000)
    assert np.isfinite(waveform).all()
