"""Independent synthetic metadata only; no video, model, network or score access."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("rank_first3", Path(__file__).parents[1] / "tools/rank_first3.py")
ranker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranker)


def protocol(module=1, metric="reprojection", direction="min"):
    return {"schema_version": 1, "frozen": True,
            "episodes": [{"id": i, "frames": (i + 1) * 10, "video_sha256": str(i) * 64} for i in range(3)],
            "groups": [{"id": "comparison", "module": module,
                        "evaluator": {"id": "synthetic", "version": "v1", "config": {"fixed": True}},
                        "upstream_fingerprint": "synthetic-upstream", "reference_fingerprint": "synthetic-baseline",
                        "budget": {"max_steps": 100},
                        "metrics": [{"id": metric, "direction": direction, "unit": "synthetic-unit"}],
                        "candidates": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}]}


def run(p, candidate="a", values=(1, 2, 3)):
    group = p["groups"][0]
    return {"group_id": group["id"], "candidate_id": candidate, "status": "complete",
            **{k: copy.deepcopy(group[k]) for k in ("evaluator", "upstream_fingerprint", "reference_fingerprint", "budget")},
            "episodes": [{**copy.deepcopy(e), "metrics": {group["metrics"][0]["id"]: value}}
                         for e, value in zip(p["episodes"], values)]}


def entries(p, runs):
    return ranker.rank_results(p, {"schema_version": 1, "runs": runs})["groups"][0]["metrics"][0]["entries"]


def a_entry(p, runs):
    return next(e for e in entries(p, runs) if e["candidate_id"] == "a")


def gate(p):
    return {"baseline_fingerprint": p["groups"][0]["reference_fingerprint"], "review_status": "passed",
            "reviewer": "Synthetic reviewer fixture",
            "evidence": [{"episode_id": e["id"], "video_sha256": e["video_sha256"],
                          "independent_video_evidence": True, "artifact": f"synthetic-review-{e['id']}.json"}
                         for e in p["episodes"]]}


def test_empty_results_leave_every_candidate_unranked():
    rows = entries(protocol(), [])
    assert all(e["status"] == "unranked" and e["rank"] is None and e["macro_mean"] is None for e in rows)
    assert all(e["reasons"] == ["no_result"] for e in rows)


@pytest.mark.parametrize("complete_result", [False, True])
def test_unfrozen_protocol_cannot_rank_any_candidate(complete_result):
    p = protocol()
    p["frozen"] = False
    rows = entries(p, [run(p)] if complete_result else [])
    assert all(e["status"] == "unranked" and e["rank"] is None and e["macro_mean"] is None for e in rows)
    assert all("protocol_not_frozen" in e["reasons"] for e in rows)


def test_protocol_must_explicitly_declare_frozen_state():
    p = protocol()
    del p["frozen"]
    with pytest.raises(ranker.ProtocolError, match="protocol.frozen"):
        entries(p, [])


def test_macro_is_equal_per_episode_and_ties_share_competition_rank():
    p = protocol()
    rows = entries(p, [run(p, "a", (0, 0, 9)), run(p, "b", (3, 3, 3)), run(p, "c", (4, 4, 4))])
    assert [(e["candidate_id"], e["macro_mean"], e["rank"]) for e in rows] == [("a", 3, 1), ("b", 3, 1), ("c", 4, 3)]


def test_max_direction_reverses_numeric_order():
    p = protocol(direction="max")
    rows = entries(p, [run(p, "a"), run(p, "b", (4, 4, 4))])
    assert [(e["candidate_id"], e["rank"]) for e in rows] == [("b", 1), ("a", 2), ("c", None)]


@pytest.mark.parametrize("status", ["failed", "skipped", "incomplete", "running", None])
def test_unsuccessful_run_cannot_rank_even_with_metrics(status):
    p = protocol()
    actual = run(p)
    actual["status"] = status
    assert a_entry(p, [actual])["status"] == "unranked"


@pytest.mark.parametrize("field,value", [("upstream_fingerprint", "other"), ("reference_fingerprint", "other"),
                                        ("evaluator", {"id": "synthetic", "version": "v2"}),
                                        ("budget", {"max_steps": 101})])
def test_incompatible_comparisons_are_unranked(field, value):
    p = protocol()
    actual = run(p)
    actual[field] = value
    row = a_entry(p, [actual])
    assert row["macro_mean"] is None and field + "_mismatch" in row["reasons"]


def test_evaluator_configuration_must_also_match():
    p = protocol()
    actual = run(p)
    actual["evaluator"]["config"]["fixed"] = 1
    assert "evaluator_mismatch" in a_entry(p, [actual])["reasons"]


def test_missing_episode_is_not_imputed():
    p = protocol()
    actual = run(p)
    actual["episodes"].pop()
    row = a_entry(p, [actual])
    assert row["macro_mean"] is None and "episode_2:missing_or_duplicate" in row["reasons"]


def test_failed_episode_is_unranked_even_if_run_claims_complete():
    p = protocol()
    actual = run(p)
    actual["episodes"][1]["status"] = "failed"
    assert "episode_1:episode_not_complete:failed" in a_entry(p, [actual])["reasons"]


@pytest.mark.parametrize("value", [None, True, "0", float("nan"), float("inf"), 10 ** 1000])
def test_missing_or_invalid_measurement_is_unranked(value):
    p = protocol()
    actual = run(p)
    actual["episodes"][0]["metrics"]["reprojection"] = value
    assert a_entry(p, [actual])["status"] == "unranked"


@pytest.mark.parametrize("field,value,reason", [("frames", 9, "frame_count_mismatch"),
                                               ("video_sha256", "f" * 64, "video_hash_mismatch")])
def test_source_video_and_full_frame_count_must_match(field, value, reason):
    p = protocol()
    actual = run(p)
    actual["episodes"][0][field] = value
    assert "episode_0:" + reason in a_entry(p, [actual])["reasons"]


def test_duplicate_episodes_and_runs_are_unranked():
    p = protocol()
    actual = run(p)
    assert "duplicate_runs" in a_entry(p, [actual, actual])["reasons"]
    actual["episodes"][2] = copy.deepcopy(actual["episodes"][0])
    assert "episode_0:missing_or_duplicate" in a_entry(p, [actual])["reasons"]


@pytest.mark.parametrize("metric", ["acceleration", "ACC-H", "ACC_O", "self_acceleration", "jerk", "smoothness"])
def test_module4_zero_acceleration_needs_fidelity_gate_even_without_flag(metric):
    p = protocol(module=4, metric=metric)
    row = a_entry(p, [run(p, values=(0, 0, 0))])
    assert row["rank"] is None and "fidelity_gate_missing" in row["reasons"]


def test_reviewed_independent_video_evidence_allows_temporal_comparison():
    p = protocol(module=4, metric="acceleration")
    actual = run(p)
    actual["fidelity_gate"] = gate(p)
    assert a_entry(p, [actual])["rank"] == 1


@pytest.mark.parametrize("change", ["baseline", "reviewer", "review_status", "evidence", "independence", "video"])
def test_inadequate_fidelity_gate_stays_unranked(change):
    p = protocol(module=4, metric="acceleration")
    actual = run(p)
    actual["fidelity_gate"] = gate(p)
    g = actual["fidelity_gate"]
    if change == "baseline":
        g["baseline_fingerprint"] = "changed"
    elif change == "reviewer":
        g["reviewer"] = ""
    elif change == "review_status":
        g["review_status"] = "pending"
    elif change == "evidence":
        g["evidence"].pop()
    elif change == "independence":
        g["evidence"][0]["independent_video_evidence"] = False
    else:
        g["evidence"][0]["video_sha256"] = "f" * 64
    assert a_entry(p, [actual])["rank"] is None


def test_groups_and_units_get_separate_tables_and_unknown_runs_are_visible():
    p = protocol()
    other = copy.deepcopy(p["groups"][0])
    other["id"] = "other-group"
    other["module"] = 2
    other["metrics"][0]["unit"] = "different-unit"
    p["groups"].append(other)
    actual = run(p)
    unknown = copy.deepcopy(actual)
    unknown["candidate_id"] = "unknown"
    report = ranker.rank_results(p, {"schema_version": 1, "runs": [actual, unknown]})
    assert len(report["groups"]) == 2
    assert all(e["status"] == "unranked" for e in report["groups"][1]["metrics"][0]["entries"])
    assert report["unrecognized_runs"][0]["status"] == "unranked"
    assert report["official_score"] is False


def test_missing_one_metric_does_not_invent_or_mix_values():
    p = protocol()
    p["groups"][0]["metrics"].append({"id": "iou", "direction": "max", "unit": "ratio"})
    report = ranker.rank_results(p, {"schema_version": 1, "runs": [run(p)]})
    metrics = report["groups"][0]["metrics"]
    assert metrics[0]["entries"][0]["rank"] == 1
    assert all(e["rank"] is None for e in metrics[1]["entries"])


def test_protocol_cannot_change_first_three_video_set():
    p = protocol()
    p["episodes"][2]["id"] = 16
    with pytest.raises(ranker.ProtocolError):
        entries(p, [])


def test_cli_empty_run_writes_json_and_markdown(tmp_path):
    p = tmp_path / "protocol.json"
    r = tmp_path / "results.json"
    out = tmp_path / "ranking.json"
    md = tmp_path / "ranking.md"
    p.write_text(json.dumps(protocol()), encoding="utf-8")
    r.write_text('{"schema_version":1,"runs":[]}', encoding="utf-8")
    assert ranker.main(["--protocol", str(p), "--results", str(r), "--output-json", str(out), "--output-md", str(md)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["official_score"] is False
    assert "unranked" in md.read_text(encoding="utf-8") and "no_result" in md.read_text(encoding="utf-8")


@pytest.mark.parametrize("raw", ['{"schema_version":1,"runs":[],"runs":[]}', '{"schema_version":NaN,"runs":[]}'])
def test_ambiguous_or_nonstandard_json_is_rejected(tmp_path, raw):
    source = tmp_path / "bad.json"
    source.write_text(raw, encoding="utf-8")
    with pytest.raises(ranker.ProtocolError):
        ranker._read_json(source)
