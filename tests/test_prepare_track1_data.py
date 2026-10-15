"""Independent synthetic fixtures for the CPU-only raw-video gate."""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("prepare_track1_data", Path(__file__).parents[1] / "tools/prepare_track1_data.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def entry(path, data=b"synthetic", lfs=False):
    result = {"type": "file", "path": f"track_1/{path}", "size": len(data),
              "oid": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
    if lfs:
        result["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
    return result


def inventory(tmp_path, entries=None):
    source = tmp_path / "inventory.json"
    source.write_text(json.dumps({"repository": gate.REPOSITORY, "revision": gate.REVISION,
                                 "files": entries or [entry(p) for p in sorted(gate.expected_paths())]}))
    return source


@pytest.mark.parametrize("path", ["../escape", "track_2/file", "C:/file", "track_1/../file", "track_1\\file", "/file", "track_1//file"])
def test_unsafe_inventory_paths_rejected(tmp_path, path):
    record = entry("README.md")
    record["path"] = path
    with pytest.raises(gate.PreparationError):
        gate.load_inventory(inventory(tmp_path, [record]))


def test_inventory_exact_coverage_and_revision(tmp_path):
    source = inventory(tmp_path)
    assert len(gate.load_inventory(source)) == 66
    doc = json.loads(source.read_text())
    doc["revision"] = "0" * 40
    source.write_text(json.dumps(doc))
    with pytest.raises(gate.PreparationError, match="pinned"):
        gate.load_inventory(source)
    doc["revision"] = gate.REVISION
    doc["files"].pop()
    source.write_text(json.dumps(doc))
    with pytest.raises(gate.PreparationError, match="missing"):
        gate.load_inventory(source)


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "bad_lfs"])
def test_inventory_rejects_bad_entries(tmp_path, mutation):
    records = [entry(p) for p in sorted(gate.expected_paths())]
    if mutation == "duplicate":
        records.append(records[0])
    elif mutation == "extra":
        records.append(entry("unapproved.bin"))
    else:
        records[0]["lfs"] = {"size": 9, "oid": "bad"}
    with pytest.raises(gate.PreparationError):
        gate.load_inventory(inventory(tmp_path, records))


@pytest.mark.parametrize("lfs", [False, True])
def test_source_hash_and_byte_size_are_verified(tmp_path, lfs):
    path = tmp_path / "source"
    path.write_bytes(b"synthetic")
    record = entry("README.md", lfs=lfs)
    result = gate.verify_file(path, record)
    assert result["sha256"] == hashlib.sha256(b"synthetic").hexdigest()
    path.write_bytes(b"changed!!")
    with pytest.raises(gate.PreparationError, match="checksum"):
        gate.verify_file(path, record)
    path.write_bytes(b"short")
    with pytest.raises(gate.PreparationError, match="byte-size"):
        gate.verify_file(path, record)


def synthetic_tree(tmp_path):
    root = tmp_path / "dataset"
    entries = {p: entry(p) for p in gate.expected_paths()}
    for name in entries:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    return root, entries


def test_tree_requires_exact_files_and_accepts_absent_optional_gitignore(tmp_path):
    root, entries = synthetic_tree(tmp_path)
    entries[".gitignore"] = entry(".gitignore")
    assert len(gate.validate_tree(root, entries)) == 66
    (root / "unapproved").mkdir()
    with pytest.raises(gate.PreparationError, match="unexpected"):
        gate.validate_tree(root, entries)


def test_tree_rejects_missing_source(tmp_path):
    root, entries = synthetic_tree(tmp_path)
    (root / "README.md").unlink()
    with pytest.raises(gate.PreparationError, match="missing"):
        gate.validate_tree(root, entries)


def test_tree_rejects_symlink_before_following_it(tmp_path):
    root, entries = synthetic_tree(tmp_path)
    target = tmp_path / "outside"
    target.write_bytes(b"synthetic")
    link = root / "README.md"
    link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not permitted on this host")
    with pytest.raises(gate.PreparationError, match="symlink"):
        gate.validate_tree(root, entries)


def frame(pts, **changes):
    values = {"pts": pts, "time_base": Fraction(1, 30), "width": 16, "height": 16, "is_corrupt": False}
    values.update(changes)
    return SimpleNamespace(**values)


def fake_av(monkeypatch, frames, declared=None):
    stream = SimpleNamespace(frames=len(frames) if declared is None else declared,
                             codec_context=SimpleNamespace(name="synthetic", options={}))

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, selected):
            assert selected is stream
            return iter(frames)

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _: Container()))


@pytest.mark.parametrize("frames,declared,match", [
    ([frame(0), frame(1, is_corrupt=True)], None, "corrupt"),
    ([frame(0), frame(1, width=32)], None, "dimensions"),
    ([frame(0), frame(None)], None, "PTS"),
    ([frame(0), frame(0)], None, "non-increasing"),
    ([frame(0), frame(1, time_base=None)], None, "time base"),
    ([frame(0), frame(1, time_base=Fraction(1, 60))], None, "time base"),
    ([frame(0)], 2, "truncated"),
    ([], 0, "empty"),
])
def test_decoder_rejects_corruption_and_invalid_timeline(monkeypatch, frames, declared, match):
    fake_av(monkeypatch, frames, declared)
    with pytest.raises(gate.PreparationError, match=match):
        gate.decode_timeline(Path("synthetic.mp4"), (16, 16))


def valid_table():
    return pd.DataFrame({"frame_index": [0, 1, 2], "episode_index": [4, 4, 4],
                         "timestamp": np.asarray([0, 1 / 30, 2 / 30], dtype=np.float32)})


def test_float32_table_timestamps_match_exact_pts(monkeypatch):
    fake_av(monkeypatch, [frame(0), frame(1), frame(2)])
    timeline = gate.decode_timeline(Path("synthetic.mp4"), (16, 16))
    assert timeline["pts"] == [0, 1, 2]
    assert timeline["frame_ids"] == [0, 1, 2]
    assert gate.validate_table(valid_table(), 4, timeline, 3)["timestamp_max_abs_error_seconds"] < 1e-6


@pytest.mark.parametrize("column,values,match", [
    ("frame_index", [0, 2, 3], "frame_index"),
    ("frame_index", [0.0, 1.0, 2.0], "noninteger"),
    ("episode_index", [4, 4, 5], "episode_index"),
    ("timestamp", [0.0, 0.1, 0.2], "disagrees"),
    ("timestamp", [0.0, float("nan"), 0.2], "nonfinite"),
    ("timestamp", [0.0, 0.0, 0.1], "non-increasing"),
])
def test_table_mismatches_rejected(column, values, match):
    table = valid_table()
    table[column] = values
    timeline = {"frame_count": 3, "timestamps_seconds": [0, 1 / 30, 2 / 30]}
    with pytest.raises(gate.PreparationError, match=match):
        gate.validate_table(table, 4, timeline, 3)


def test_table_count_must_equal_decode_and_metadata():
    with pytest.raises(gate.PreparationError, match="frame-count"):
        gate.validate_table(valid_table(), 4, {"frame_count": 2}, 3)


def test_real_synthetic_video_decodes_all_frames(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "synthetic.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(3):
            video_frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 50, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    timeline = gate.decode_timeline(path, (16, 16))
    assert timeline["frame_count"] == 3
    gate.validate_table(valid_table(), 4, timeline, 3)


def test_existing_output_never_overwritten(tmp_path, monkeypatch):
    fake_av(monkeypatch, [])
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(gate.PreparationError, match="already exists"):
        gate.prepare(tmp_path / "source", tmp_path / "inventory.json", output)
    assert sentinel.read_text() == "keep"


def test_output_overlap_rejected_after_normalizing_parent_segments(tmp_path, monkeypatch):
    fake_av(monkeypatch, [])
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(gate.PreparationError, match="overlap"):
        gate.prepare(root, tmp_path / "inventory.json", root / "nested" / ".." / "output")


def test_full_synthetic_handoff_stays_raw_only_and_hashes_outputs(tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    root, _ = synthetic_tree(tmp_path)
    video = tmp_path / "independent.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i in range(3):
            sample = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 40, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(sample):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    for index in gate.EPISODE_IDS:
        video_path, table_path = gate.episode_paths(index)
        (root / video_path).write_bytes(video.read_bytes())
        table = valid_table()
        table["episode_index"] = index
        table.to_parquet(root / table_path)
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": 30, "total_videos": 30,
        "total_frames": 90, "fps": 30, "features": {gate.VIDEO_KEY: {"shape": [16, 16, 3]}}}))
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "length": 3}) for i in range(30)))
    (root / "meta/episodes_metadata.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "video_key": gate.VIDEO_KEY}) for i in range(30)))
    source = inventory(tmp_path, [entry(p, (root / p).read_bytes(), lfs=p.endswith((".mp4", ".parquet")))
                                  for p in sorted(gate.expected_paths())])
    output = tmp_path / "prepared"
    manifest = gate.prepare(root, source, output)
    assert manifest["total_frames"] == 90
    assert manifest["raw_data_ready"] is True
    assert manifest["body_inference_ready"] is False
    assert manifest["models_executed"] is False
    assert len(manifest["episodes"]) == 30
    checks = json.loads((output / "checks.json").read_text())
    assert checks["passed"] is True
    assert checks["manifest_sha256"] == gate.sha256(output / "manifest.json")
    hashes = (output / "SHA256SUMS").read_text().splitlines()
    assert len(hashes) == 32
    for line in hashes:
        digest, relative = line.split("  ", 1)
        assert gate.sha256(output / relative) == digest
    original_decode = gate.decode_timeline
    original_readme = (root / "README.md").read_bytes()
    original_inventory = source.read_bytes()
    for mutation in ["source", "inventory"]:
        def mutate_during_decode(*args):
            timeline = original_decode(*args)
            if mutation == "source":
                (root / "README.md").write_bytes(b"changed!!")
            else:
                source.write_bytes(original_inventory + b"\n")
            return timeline
        monkeypatch.setattr(gate, "decode_timeline", mutate_during_decode)
        altered = tmp_path / f"changed-{mutation}"
        with pytest.raises(gate.PreparationError, match="checksum mismatch|inventory changed"):
            gate.prepare(root, source, altered)
        assert not (altered / "manifest.json").exists()
        assert json.loads((altered / "checks.json").read_text())["raw_data_ready"] is False
        (root / "README.md").write_bytes(original_readme)
        source.write_bytes(original_inventory)
    # A failed subsequent preparation must never expose a ready manifest.
    def fail_decode(*_):
        raise gate.PreparationError("synthetic corrupt decode")
    monkeypatch.setattr(gate, "decode_timeline", fail_decode)
    failed = tmp_path / "failed"
    with pytest.raises(gate.PreparationError, match="corrupt"):
        gate.prepare(root, source, failed)
    assert not (failed / "manifest.json").exists()
    failed_checks = json.loads((failed / "checks.json").read_text())
    assert failed_checks["raw_data_ready"] is False
    assert failed_checks["passed"] is False
