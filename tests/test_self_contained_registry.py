from pathlib import Path
import sys

import pytest


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "self-contained"))
import registry


def test_model_spec_pins_large_checkpoint_and_native_audio() -> None:
    spec = registry.model_spec()
    assert spec.source_revision == "b5786c5dc0ae8f766337fdc1b67ab6046586d14d"
    assert spec.name == "audioldm2-full-large-1150k"
    assert spec.sample_rate == 16000
    assert spec.duration_seconds == 10.0
    assert spec.checkpoint.repo_id == "haoheliu/audioldm2-full-large-1150k"
    assert spec.checkpoint.revision == "327bb3e09b48caeed6c283cfe16cdb2323dfb374"
    assert spec.checkpoint.size_bytes == 11470645003
    assert spec.checkpoint.sha256 == (
        "6b483a47480d15d90c2eaaa17415484538bff7c6d9d2282f7838ad004c637c09"
    )


def test_validate_artifact_rejects_wrong_size(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pth"
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        registry.validate_artifact(artifact, registry.model_spec().checkpoint)


def test_validate_artifact_rejects_wrong_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pth"
    artifact.write_bytes(b"same-size")
    expected = registry.ArtifactSpec(
        repo_id="test/repo",
        revision="abc",
        filename="checkpoint.pth",
        size_bytes=artifact.stat().st_size,
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        registry.validate_artifact(artifact, expected)


def test_auxiliary_assets_have_local_names_used_by_vendor() -> None:
    assets = {
        item.local_name: item for item in registry.model_spec().auxiliary_assets
    }
    assert set(assets) == {"google/flan-t5-large", "roberta-base", "gpt2"}
    assert assets["google/flan-t5-large"].revision == (
        "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    assert assets["roberta-base"].revision == (
        "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
    )
    assert assets["gpt2"].revision == (
        "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )


def test_requirements_are_exactly_pinned() -> None:
    requirements = root / "self-contained" / "requirements.txt"
    entries = [
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert entries
    assert all(line.count("==") == 1 for line in entries)
    assert all(
        not any(operator in line for operator in (">=", "<=", "~=", "!="))
        for line in entries
    )
