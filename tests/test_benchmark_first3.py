"""Independent synthetic files only; never run a reconstruction model."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("first3", Path(__file__).parents[1] / "tools/benchmark_first3.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "benchmarks/first3").mkdir(parents=True)
    (repo / "tools").mkdir()
    for name in ("benchmark_first3.py", "benchmark_moge_video.py", "rank_first3.py"):
        (repo / "tools" / name).write_text("# synthetic\n")
    (repo / "benchmarks/first3/README.md").write_text("Synthetic kit")
    dataset = tmp_path / "track_1"
    (dataset / "meta").mkdir(parents=True)
    bench.write(dataset / "meta/info.json", {"robot_type": "video_only_object_tracking"})
    eps = []
    for i in range(3):
        path = dataset / f"video{i}.mp4"
        path.write_bytes(f"synthetic-video-{i}".encode())
        eps.append({"id": i, "frames": i + 4, "bytes": path.stat().st_size,
                    "video_sha256": bench.digest(path), "source_path": path.name,
                    "video": f"videos/episode_{i:06d}.0.color.mp4"})
    bench.write(repo / "benchmarks/first3/episodes.json", {"dataset": "track1", "episodes": eps})
    for name in ("episodes.jsonl", "episodes_metadata.jsonl"):
        (dataset / "meta" / name).write_text("".join(json.dumps({"episode_index": i}) + "\n" for i in range(4)))
    monkeypatch.setattr(bench, "__file__", str(repo / "tools/benchmark_first3.py"))
    return repo, dataset, tmp_path / "package"


def test_preparation_copies_exact_subset_and_keeps_results_empty(tmp_path, monkeypatch):
    _, dataset, output = fixture(tmp_path, monkeypatch)
    bench.prepare(dataset, output)
    assert len(bench.verify_videos(output)) == 3
    assert bench.read(output / "results.json")["runs"] == []
    assert len((output / "metadata/episodes.jsonl").read_text().splitlines()) == 3
    for entry in bench.read(output / "package-manifest.json")["files"]:
        assert bench.digest(output / entry["path"]) == entry["sha256"]
    with pytest.raises(FileExistsError):
        bench.prepare(dataset, output)


def test_changed_video_rejected_before_output_is_created(tmp_path, monkeypatch):
    _, dataset, output = fixture(tmp_path, monkeypatch)
    (dataset / "video1.mp4").write_bytes(b"different video")
    with pytest.raises(ValueError, match="differs"):
        bench.prepare(dataset, output)
    assert not output.exists()


def test_plan_does_not_run_or_create_outputs_and_resolves_artifacts(tmp_path, monkeypatch):
    repo, dataset, package = fixture(tmp_path, monkeypatch)
    bench.write(repo / "benchmarks/first3/test_recipes.json", {"recipes": [{"id": "test",
        "required_paths": [{"path": "{shared}/missing", "kind": "file"}],
        "steps": [{"name": "never_run", "argv": ["{python}", "--do-not-execute"], "cwd": "{package}"}],
        "expected_outputs": ["predictions/*.npz"]}]})
    bench.prepare(dataset, package)
    target = tmp_path / "runs"
    plan = bench.make_plan(package, {"python": "does-not-exist", "assets": str(tmp_path),
        "toolkit": str(tmp_path), "shared": str(tmp_path)}, "test", target)
    assert plan["status"] == "missing_inputs"
    assert len(plan["jobs"]) == 3
    assert plan["jobs"][0]["expected_outputs"] == [str(target / "test/episode_000000/predictions/*.npz")]
    assert not target.exists()


def test_path_escape_and_track2_rejected(tmp_path, monkeypatch):
    _, dataset, output = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="leaves"):
        bench.checked_child(dataset, "../escape")
    with pytest.raises(ValueError, match="Track 2"):
        bench.prepare(tmp_path / "track_2", output)
