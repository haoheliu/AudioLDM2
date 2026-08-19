from pathlib import Path
import os
import subprocess
import sys

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
            sys.executable,
            "self-contained/run_inference.py",
            "--prompt",
            "A gentle rainstorm outside a quiet cabin",
            "--output",
            str(output),
            "--cache-dir",
            str(root / ".cache" / "audioldm2"),
            "--device",
            "cuda",
            "--seed",
            "0",
            "--ddim-steps",
            "2",
            "--guidance-scale",
            "3.5",
            "--candidates",
            "1",
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
