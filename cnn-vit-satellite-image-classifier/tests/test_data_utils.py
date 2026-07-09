"""Tests for src/data_utils.py."""

import tarfile
from pathlib import Path

from src.data_utils import extract_tar_archive, list_image_files, summarize_class_distribution


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real image, just needs to exist")


def test_list_image_files_finds_only_known_extensions_and_sorts(tmp_path):
    _touch(tmp_path / "class_a" / "b.png")
    _touch(tmp_path / "class_a" / "a.jpg")
    _touch(tmp_path / "class_b" / "c.JPEG")
    _touch(tmp_path / "class_b" / "readme.txt")

    files = list_image_files(tmp_path)

    assert [f.name for f in files] == ["a.jpg", "b.png", "c.JPEG"]


def test_list_image_files_ignores_directories_and_missing_extensions(tmp_path):
    (tmp_path / "class_a").mkdir()
    (tmp_path / "class_a" / "subdir.png").mkdir()

    assert list_image_files(tmp_path) == []


def test_summarize_class_distribution_counts_per_parent_folder(tmp_path):
    _touch(tmp_path / "class_0_non_agri" / "img1.jpg")
    _touch(tmp_path / "class_0_non_agri" / "img2.jpg")
    _touch(tmp_path / "class_1_agri" / "img3.png")

    counts = summarize_class_distribution(tmp_path)

    assert counts == {"class_0_non_agri": 2, "class_1_agri": 1}


def test_summarize_class_distribution_empty_dir_returns_empty_dict(tmp_path):
    assert summarize_class_distribution(tmp_path) == {}


def test_extract_tar_archive_creates_output_dir_and_extracts_members(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    member_path = source_dir / "class_1_agri" / "img.jpg"
    _touch(member_path)

    archive_path = tmp_path / "dataset.tar"
    with tarfile.open(archive_path, "w") as tar:
        tar.add(source_dir, arcname=".")

    output_dir = tmp_path / "extracted" / "nested"
    result = extract_tar_archive(archive_path, output_dir)

    assert result == output_dir
    assert (output_dir / "class_1_agri" / "img.jpg").exists()
