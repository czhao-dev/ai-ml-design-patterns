"""Reusable helpers for local aircraft damage image data."""

from __future__ import annotations

import tarfile
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_image_files(dataset_dir: str | Path) -> list[Path]:
    """Return image files below a directory, sorted for reproducible iteration."""
    root = Path(dataset_dir)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def summarize_class_distribution(dataset_dir: str | Path) -> dict[str, int]:
    """Count images by class folder name."""
    counts: Counter[str] = Counter()
    for image_path in list_image_files(dataset_dir):
        counts[image_path.parent.name] += 1
    return dict(sorted(counts.items()))


def extract_tar_archive(archive_path: str | Path, output_dir: str | Path) -> Path:
    """Extract a tar archive and return the output directory."""
    archive = Path(archive_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive) as tar_ref:
        tar_ref.extractall(destination)

    return destination
