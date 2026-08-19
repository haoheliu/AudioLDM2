from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import soundfile as sf
import torch

from generator import AudioLDM2Generator
from registry import (
    ModelSpec,
    model_spec,
    prefetch_auxiliary_assets,
    resolve_checkpoint,
)


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
    if waveform.ndim != 3 or waveform.shape[:2] != (1, 1):
        raise ValueError(
            f"expected [1, 1, samples], got {waveform.shape}"
        )
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains non-finite samples")
    minimum_samples = int(spec.sample_rate * (spec.duration_seconds - 0.5))
    maximum_samples = int(spec.sample_rate * (spec.duration_seconds + 0.25))
    if not minimum_samples <= waveform.shape[-1] <= maximum_samples:
        raise ValueError(
            f"waveform duration is outside the fixed protocol: "
            f"{waveform.shape[-1]} samples"
        )


def write_waveform_exclusive(
    output: Path,
    waveform: np.ndarray,
    sample_rate: int,
) -> None:
    created = False
    try:
        with output.open("xb") as handle:
            created = True
            sf.write(
                handle,
                waveform,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
    except FileExistsError as error:
        raise FileExistsError(f"output already exists: {output}") from error
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise


def run_inference(config: RunConfig) -> Path:
    if config.output.exists():
        raise FileExistsError(f"output already exists: {config.output}")
    spec = model_spec()
    device = choose_device(config.device)
    asset_root = config.cache_dir / "assets"
    prefetch_auxiliary_assets(spec, asset_root)
    checkpoint = resolve_checkpoint(spec, config.cache_dir / "huggingface")
    generator = AudioLDM2Generator(
        spec,
        checkpoint,
        asset_root,
        device,
    )
    waveform = generator.generate(
        config.prompt,
        config.seed,
        config.ddim_steps,
        config.guidance_scale,
        config.candidates,
    )
    validate_waveform(waveform, spec)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    write_waveform_exclusive(
        config.output,
        waveform[0, 0],
        spec.sample_rate,
    )
    return config.output


@click.command()
@click.option("--prompt", required=True, type=str)
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".cache/audioldm2"),
    show_default=True,
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "cpu"]),
    default="auto",
    show_default=True,
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--ddim-steps",
    type=click.IntRange(min=1),
    default=200,
    show_default=True,
)
@click.option(
    "--guidance-scale",
    type=click.FloatRange(min=0.0),
    default=3.5,
    show_default=True,
)
@click.option(
    "--candidates",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
)
def main(
    prompt: str,
    output: Path,
    cache_dir: Path,
    device: str,
    seed: int,
    ddim_steps: int,
    guidance_scale: float,
    candidates: int,
) -> None:
    config = RunConfig(
        prompt=prompt,
        output=output,
        cache_dir=cache_dir,
        device=device,
        seed=seed,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        candidates=candidates,
    )
    try:
        result = run_inference(config)
    except (FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote {result}")


if __name__ == "__main__":
    main()
