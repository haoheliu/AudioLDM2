from pathlib import Path
import subprocess


root = Path(__file__).resolve().parents[1]
vendor_root = root / "self-contained" / "vendor" / "audioldm2"
source_revision = "b5786c5dc0ae8f766337fdc1b67ab6046586d14d"


def tracked_package_files() -> list[Path]:
    output = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            source_revision,
            "--",
            "audioldm2",
        ],
        cwd=root,
        text=True,
    )
    return [Path(line) for line in output.splitlines() if line]


def committed_bytes(path: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{source_revision}:{path.as_posix()}"],
        cwd=root,
    )


def test_vendor_tree_matches_every_tracked_package_file() -> None:
    tracked = tracked_package_files()
    expected = {path.relative_to("audioldm2") for path in tracked}
    actual = {
        path.relative_to(vendor_root)
        for path in vendor_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert actual == expected
    for relative_path in sorted(expected):
        assert (vendor_root / relative_path).read_bytes() == committed_bytes(
            Path("audioldm2") / relative_path
        )


def test_vendor_license_is_verbatim() -> None:
    assert (
        root / "self-contained" / "vendor" / "LICENSE"
    ).read_bytes() == committed_bytes(Path("LICENSE"))
