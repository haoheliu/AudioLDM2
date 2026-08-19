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

    return VendorApi(
        latent_diffusion_type=LatentDiffusion,
        default_config=default_audioldm_config,
        make_batch=make_batch_for_text_to_audio,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AudioLDM2Generator:
    def __init__(
        self,
        spec: ModelSpec,
        checkpoint_path: Path,
        asset_root: Path,
        device: str,
        api: VendorApi | None = None,
    ):
        self.spec = spec
        self.device = torch.device(device)
        vendor_root = Path(__file__).resolve().parent / "vendor"
        with working_directory(asset_root):
            self.api = api or load_vendor_api(vendor_root)
            config = self.api.default_config(spec.name)
            config["model"]["params"]["device"] = self.device
            self.model = self.api.latent_diffusion_type(
                **config["model"]["params"]
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if "state_dict" not in checkpoint:
            raise ValueError(f"checkpoint has no state_dict: {checkpoint_path}")
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        gc.collect()
        self.model.eval().to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: int,
        ddim_steps: int,
        guidance_scale: float,
        candidates: int,
    ) -> np.ndarray:
        seed_everything(seed)
        batch = self.api.make_batch(
            prompt,
            transcription="",
            batchsize=1,
        )
        self.model.latent_t_size = int(
            self.spec.duration_seconds * self.spec.latent_t_per_second
        )
        waveform = self.model.generate_batch(
            batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_gen=candidates,
            duration=self.spec.duration_seconds,
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"invalid waveform tensor: shape={waveform.shape}")
        if not np.isfinite(waveform).all():
            raise ValueError("waveform contains non-finite samples")
        return waveform
