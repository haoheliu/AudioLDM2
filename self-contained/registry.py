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
    source_revision: str
    sample_rate: int
    duration_seconds: float
    latent_t_per_second: float
    default_ddim_steps: int
    default_guidance_scale: float
    default_candidates: int
    checkpoint: ArtifactSpec
    auxiliary_assets: tuple[ArtifactSpec, ...]


def checkpoint_spec() -> ArtifactSpec:
    return ArtifactSpec(
        repo_id="haoheliu/audioldm2-full-large-1150k",
        revision="327bb3e09b48caeed6c283cfe16cdb2323dfb374",
        filename="audioldm2-full-large-1150k.pth",
        size_bytes=11470645003,
        sha256="6b483a47480d15d90c2eaaa17415484538bff7c6d9d2282f7838ad004c637c09",
    )


def auxiliary_specs() -> tuple[ArtifactSpec, ...]:
    flan_t5 = ArtifactSpec(
        repo_id="google/flan-t5-large",
        revision="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
        local_name="google/flan-t5-large",
        allow_patterns=(
            "config.json",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
    )
    roberta = ArtifactSpec(
        repo_id="FacebookAI/roberta-base",
        revision="e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        local_name="roberta-base",
        allow_patterns=(
            "config.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    )
    gpt2 = ArtifactSpec(
        repo_id="openai-community/gpt2",
        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        local_name="gpt2",
        allow_patterns=("config.json",),
    )
    return flan_t5, roberta, gpt2


def model_spec() -> ModelSpec:
    return ModelSpec(
        name="audioldm2-full-large-1150k",
        source_revision="b5786c5dc0ae8f766337fdc1b67ab6046586d14d",
        sample_rate=16000,
        duration_seconds=10.0,
        latent_t_per_second=25.6,
        default_ddim_steps=200,
        default_guidance_scale=3.5,
        default_candidates=3,
        checkpoint=checkpoint_spec(),
        auxiliary_assets=auxiliary_specs(),
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
    path = Path(
        hf_hub_download(
            repo_id=spec.checkpoint.repo_id,
            filename=spec.checkpoint.filename,
            revision=spec.checkpoint.revision,
            cache_dir=cache_dir,
        )
    )
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
