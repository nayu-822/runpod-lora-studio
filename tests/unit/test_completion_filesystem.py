from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from runpod_lora_studio.services.completion_filesystem import (
    CompletionFilesystemError,
    SafeExportDirectory,
)


def test_safe_export_directory_round_trip_on_supported_fallback(
    test_workspace: Path,
) -> None:
    projects = test_workspace / "projects"
    with SafeExportDirectory.open_root(
        projects, ("project", "training", "exports", "job")
    ) as export:
        export.write_json("provenance/source.json", {"source": "test"})
        assert export.read_bytes("provenance/source.json") == b'{"source":"test"}\n'
        assert export.regular_files() == [
            (
                "provenance/source.json",
                len(b'{"source":"test"}\n'),
                hashlib.sha256(b'{"source":"test"}\n').hexdigest(),
            )
        ]


def test_safe_export_directory_rejects_symlink_component(test_workspace: Path) -> None:
    projects = test_workspace / "projects"
    projects.mkdir()
    outside = test_workspace / "outside"
    outside.mkdir()
    project = projects / "project"
    try:
        project.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises((CompletionFilesystemError, OSError)):
        SafeExportDirectory.open_root(
            projects, ("project", "training", "exports"), create_missing=True
        )
    assert not (outside / "training").exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows locks the open source file")
def test_safe_export_copy_detects_source_replacement(test_workspace: Path) -> None:
    projects = test_workspace / "projects"
    source = test_workspace / "source.bin"
    source.write_bytes(b"original-source")
    swapped = False
    with SafeExportDirectory.open_root(
        projects, ("project", "training", "exports", "job")
    ) as export:

        def replace_source() -> None:
            nonlocal swapped
            if not swapped:
                swapped = True
                source.unlink()
                source.write_bytes(b"replacement-source-with-a-different-size")

        with pytest.raises(CompletionFilesystemError):
            export.copy_file(
                source, "artifacts/model.safetensors", cancel=replace_source
            )
        assert export.child_exists("artifacts") == "directory"
        assert export.child_exists("artifacts/model.safetensors") == "missing"


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory fd semantics")
def test_cleanup_never_removes_a_replaced_creating_directory(
    test_workspace: Path,
) -> None:
    projects = test_workspace / "projects"
    with SafeExportDirectory.open_root(
        projects, ("project", "training", "exports")
    ) as exports:
        creating = exports.open_child(".creating-owned", create=True)
        creating.write_bytes_atomic("owned.txt", b"owned")
        replacement_path = creating.path
        shutil.rmtree(replacement_path)
        replacement_path.mkdir()
        (replacement_path / "active.txt").write_bytes(b"active")

        with pytest.raises(CompletionFilesystemError):
            exports.remove_child(creating)
        assert (replacement_path / "active.txt").read_bytes() == b"active"
        creating.close()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory fd semantics")
def test_atomic_rename_rejects_existing_symlink_destination(
    test_workspace: Path,
) -> None:
    projects = test_workspace / "projects"
    outside = test_workspace / "outside"
    outside.mkdir()
    with SafeExportDirectory.open_root(
        projects, ("project", "training", "exports")
    ) as exports:
        creating = exports.open_child(".creating-owned", create=True)
        creating.write_bytes_atomic("marker.json", b"owned")
        (exports.path / "job").symlink_to(outside, target_is_directory=True)
        with pytest.raises(CompletionFilesystemError):
            exports.rename_child(creating, "job")
        creating.close()
        assert not (outside / "marker.json").exists()
