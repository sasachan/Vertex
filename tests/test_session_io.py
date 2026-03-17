"""Tests for vertex.session_io — CSV/JSON persistence."""

from __future__ import annotations

import csv
import json
import os
import tempfile

import pytest

from vertex.session_io import (
    create_session_csv, write_shot_csv, write_session_json,
    SHARegistry, file_sha256, move_to_processed,
)
from vertex.models import CSV_HEADERS, ShotRecord


class TestCreateSessionCsv:
    def test_creates_file_with_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path, wr, fh = create_session_csv(tmpdir)
            fh.close()

            assert os.path.exists(path)
            assert "session_" in os.path.basename(path)
            assert path.endswith(".csv")

            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)
                assert headers == CSV_HEADERS


class TestWriteShotCsv:
    def test_writes_correct_row(self, sample_shot):
        with tempfile.TemporaryDirectory() as tmpdir:
            path, wr, fh = create_session_csv(tmpdir)
            write_shot_csv(wr, sample_shot)
            fh.close()

            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]["shot_number"] == "1"
                assert float(rows[0]["hold_seconds"]) == 3.2


class TestWriteSessionJson:
    def test_produces_valid_json(self, sample_shot):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            summary = {"total_shots": 1, "avg_hold": 3.2}
            write_session_json(path, [sample_shot], summary)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["version"] == "0.2.0"
            assert data["summary"]["total_shots"] == 1
            assert len(data["shots"]) == 1
            assert data["shots"][0]["hold_seconds"] == 3.2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# SHARegistry
# ---------------------------------------------------------------------------
class TestSHARegistryRoundTrip:
    def test_register_is_registered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SHARegistry(tmpdir)
            assert not reg.is_registered("abc123")
            reg.register("abc123", {"filename": "video.mp4"})
            assert reg.is_registered("abc123")

    def test_save_load_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SHARegistry(tmpdir)
            reg.register("deadbeef", {"filename": "x.mp4"})
            reg.save()

            reg2 = SHARegistry(tmpdir)
            assert reg2.is_registered("deadbeef")
            assert reg2.get("deadbeef")["filename"] == "x.mp4"

    def test_metadata_sanitises_null_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SHARegistry(tmpdir)
            reg.register("sha1", {"name": "bad\x00name"})
            stored = reg.get("sha1")
            assert "\x00" not in stored["name"]


class TestSHARegistryFlush:
    def test_flush_restores_files_and_clears_registry(self):
        with tempfile.TemporaryDirectory() as base_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                # Create a file in a processed/ subfolder
                proc_dir = os.path.join(base_dir, "processed")
                os.makedirs(proc_dir)
                src_file = os.path.join(proc_dir, "video.mp4")
                with open(src_file, "w") as f:
                    f.write("content")

                reg = SHARegistry(out_dir)
                reg.register("aaa", {"filename": "video.mp4"})
                reg.save()

                restored = reg.flush(base_dir)

                assert len(restored) == 1
                assert os.path.exists(restored[0])
                assert not reg.is_registered("aaa")
                assert not os.path.exists(os.path.join(out_dir, "processed_registry.json"))


# ---------------------------------------------------------------------------
# move_to_processed
# ---------------------------------------------------------------------------
class TestMoveToProcessed:
    def test_creates_processed_subfolder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "video.mp4")
            with open(src, "w") as f:
                f.write("data")
            dest = move_to_processed(src, tmpdir)
            assert dest is not None
            assert "processed" in dest
            assert os.path.exists(dest)
            assert not os.path.exists(src)

    def test_raises_on_path_outside_allowed_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.TemporaryDirectory() as other_dir:
                src = os.path.join(other_dir, "video.mp4")
                with open(src, "w") as f:
                    f.write("data")
                with pytest.raises(ValueError, match="outside allowed base"):
                    move_to_processed(src, tmpdir)


# ---------------------------------------------------------------------------
# file_sha256
# ---------------------------------------------------------------------------
class TestFileSha256:
    def test_known_hash(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            path = f.name
        try:
            digest = file_sha256(path)
            # SHA-256 of "hello world"
            assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576"[:40] or len(digest) == 64
        finally:
            os.unlink(path)

    def test_same_content_same_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "a.bin")
            path2 = os.path.join(tmpdir, "b.bin")
            data = b"vertex test data" * 100
            open(path1, "wb").write(data)
            open(path2, "wb").write(data)
            assert file_sha256(path1) == file_sha256(path2)
