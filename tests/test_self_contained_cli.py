from pathlib import Path
import sys

import numpy as np
import pytest
from click.testing import CliRunner


root = Path(__file__).resolve().parents[1]
self_contained = root / "self-contained"
sys.path.insert(0, str(self_contained))
import run_inference


def test_cli_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "exists.wav"
    output.write_bytes(b"keep")
    result = CliRunner().invoke(
        run_inference.main,
        [
            "--prompt",
            "rain",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_bytes() == b"keep"


def test_validate_waveform_rejects_nonfinite_audio() -> None:
    waveform = np.array([[[np.nan, 0.0]]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        run_inference.validate_waveform(waveform, run_inference.model_spec())


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 160000),
        (1, 1, 320000),
    ],
)
def test_validate_waveform_enforces_fixed_output_protocol(shape) -> None:
    waveform = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError):
        run_inference.validate_waveform(waveform, run_inference.model_spec())


def test_choose_device_rejects_unavailable_cuda(monkeypatch) -> None:
    monkeypatch.setattr(run_inference.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA was requested"):
        run_inference.choose_device("cuda")


def test_late_output_race_does_not_overwrite_competing_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "race.wav"

    class FakeGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            output.write_bytes(b"competitor")
            return np.zeros((1, 1, 160000), dtype=np.float32)

    monkeypatch.setattr(
        run_inference,
        "prefetch_auxiliary_assets",
        lambda *args: None,
    )
    monkeypatch.setattr(
        run_inference,
        "resolve_checkpoint",
        lambda *args: tmp_path / "model.pth",
    )
    monkeypatch.setattr(run_inference, "AudioLDM2Generator", FakeGenerator)
    config = run_inference.RunConfig(
        prompt="rain",
        output=output,
        cache_dir=tmp_path / "cache",
        device="cpu",
        seed=0,
        ddim_steps=2,
        guidance_scale=3.5,
        candidates=1,
    )
    with pytest.raises(FileExistsError):
        run_inference.run_inference(config)
    assert output.read_bytes() == b"competitor"
