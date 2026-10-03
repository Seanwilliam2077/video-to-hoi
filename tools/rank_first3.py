#!/usr/bin/env python3
"""Rank declared Track 1 diagnostics, separately for each group and metric.

This stdlib-only tool reads JSON metadata; it does not run models, inspect videos,
verify measurement/reviewer claims, or compute official competition scores.
Empty results are valid and leave every candidate unranked. No value is imputed.

Protocol schema (schema_version=1)::

    {"schema_version": 1, "frozen": false,
     "episodes": [{"id": 0, "frames": 100, "video_sha256": "<64 hex>"},
                  {"id": 1, "frames": 200, "video_sha256": "<64 hex>"},
                  {"id": 2, "frames": 300, "video_sha256": "<64 hex>"}],
     "groups": [{"id": "module4", "module": 4,
       "evaluator": {"id": "diagnostics", "version": "v1", "config": {}},
       "upstream_fingerprint": "frozen-input-snapshot",
       "reference_fingerprint": "frozen-baseline-snapshot",
       "budget": {"max_steps": 300},
       "metrics": [{"id": "self_acceleration", "direction": "min",
                    "definition": "mean_norm_prediction_second_difference_over_dt_squared",
                    "unit": "m/s^2", "requires_fidelity_gate": true}],
       "candidates": [{"id": "raw"}, {"id": "smoothnet"}]}]}

Results start as {"schema_version": 1, "runs": []}. Each run records group_id,
candidate_id, status="complete", exact copies of evaluator (including config),
upstream_fingerprint, reference_fingerprint and budget, and all three episodes.
Each run also declares provenance={"dataset": "track1", "output_kind":
"reconstruction", "contains_fake_outputs": false}. Missing declarations,
fake/synthetic outputs and explicit official-score claims remain unranked.
These declarations are not proof that an output was produced by a real model.
Each episode repeats id/frames/video_sha256 and adds metrics={metric_id: number}.
Duplicate runs are unranked; choose one run before invoking this tool.
The protocol must declare frozen=true only after its machine budget, upstream,
reference and independent observations are fixed. With frozen=false every
candidate remains unranked, including a run that claims complete measurements.

Gated runs additionally declare fidelity_gate with baseline_fingerprint equal
to reference_fingerprint, review_status="passed", a named reviewer, and evidence
for EACH episode: {episode_id, video_sha256, independent_video_evidence: true,
artifact: "path-or-URL-of-video-evidence"}. Evidence must establish fidelity to
the original video; a smoother trajectory alone is insufficient. Module 4
prediction-only self_acceleration/jerk/smoothness metrics require this gate, even if the
protocol omits the flag. Other metrics may opt in with requires_fidelity_gate.
Official metric names (including ACC-H/O, CD-H/O and PEN) are not accepted by
this no-reference diagnostic ranker. self_acceleration is NOT official ACC:
it measures prediction curvature, not prediction/reference acceleration error.
The old ambiguous acceleration key is rejected, never silently converted.

Values are macro-averaged with equal weight for episodes 0, 1 and 2. Exact ties
receive competition ranks (1, 1, 3). Units and modules are never combined.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
from typing import Any


class ProtocolError(ValueError):
    """The requested comparison is ambiguous or malformed."""


SELF_ACCELERATION_DEFINITION = "mean_norm_prediction_second_difference_over_dt_squared"


def _official_claim(value: Any) -> bool:
    """Catch explicit official-score claims, including inside evaluator config."""
    if isinstance(value, dict):
        return any((key == "official_score" and item is not False) or _official_claim(item)
                   for key, item in value.items())
    return isinstance(value, list) and any(_official_claim(item) for item in value)


def _non_reconstruction_claim(value: Any) -> bool:
    """Reject explicit smoke/fake markers; absent or falsified markers are not detectable."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"fake", "synthetic", "contains_fake_outputs"} and item is not False:
                return True
            if key == "output_kind" and item != "reconstruction":
                return True
            if key == "backend" and isinstance(item, str) and re.search(r"(^|[._-])(fake|synthetic|smoke)([._-]|$)", item.lower()):
                return True
            if _non_reconstruction_claim(item):
                return True
    return isinstance(value, list) and any(_non_reconstruction_claim(item) for item in value)


def _validate_metric(metric: dict) -> None:
    name = re.sub(r"[^a-z0-9]+", "_", metric["id"].lower()).strip("_")
    if re.search(r"^(cd|pen)(_|$)", name) or name.startswith("official_") or name in {
        "total_score", "overall_score", "combined_score", "composite_score", "final_score"
    }:
        raise ProtocolError("official metric names are not supported by this diagnostic ranker")
    acceleration = re.search(r"(^|_)(acc|acceleration)(_|$)", name)
    if acceleration and metric["id"] != "self_acceleration":
        raise ProtocolError("use the explicitly defined self_acceleration diagnostic; official ACC is reference-relative")
    if metric["id"] == "self_acceleration":
        if metric.get("definition") != SELF_ACCELERATION_DEFINITION or metric.get("unit") != "m/s^2":
            raise ProtocolError("self_acceleration requires its prediction-only definition and m/s^2 units")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _same(left: Any, right: Any) -> bool:
    """Compare metadata without Python's True == 1 equivalence."""
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _validate_protocol(protocol: Any) -> None:
    if not isinstance(protocol, dict) or not _same(protocol.get("schema_version"), 1):
        raise ProtocolError("protocol.schema_version must be 1")
    if not isinstance(protocol.get("frozen"), bool):
        raise ProtocolError("protocol.frozen must explicitly be true or false")
    if _official_claim(protocol):
        raise ProtocolError("this protocol cannot claim official scores")
    episodes = protocol.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 3:
        raise ProtocolError("protocol requires exactly episodes 0, 1, 2")
    for episode in episodes:
        if not isinstance(episode, dict) or not _integer(episode.get("id")):
            raise ProtocolError("episode id must be an integer")
        if not _integer(episode.get("frames")) or episode["frames"] <= 0 or not _sha(episode.get("video_sha256")):
            raise ProtocolError("each episode needs positive frames and a video SHA-256")
    if {episode["id"] for episode in episodes} != {0, 1, 2}:
        raise ProtocolError("protocol requires exactly episodes 0, 1, 2")
    groups = protocol.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ProtocolError("protocol.groups must be a nonempty list")
    group_ids = set()
    for group in groups:
        if not isinstance(group, dict) or not _text(group.get("id")) or group["id"] in group_ids:
            raise ProtocolError("group ids must be nonempty and unique")
        group_ids.add(group["id"])
        if not _integer(group.get("module")) or group["module"] not in (1, 2, 4):
            raise ProtocolError("group.module must be 1, 2, or 4")
        evaluator = group.get("evaluator")
        if not isinstance(evaluator, dict) or not all(_text(evaluator.get(k)) for k in ("id", "version")):
            raise ProtocolError("group.evaluator requires id and version")
        if not all(_text(group.get(k)) for k in ("upstream_fingerprint", "reference_fingerprint")):
            raise ProtocolError("group requires frozen upstream and reference fingerprints")
        if not isinstance(group.get("budget"), dict) or not group["budget"]:
            raise ProtocolError("group.budget must explicitly declare a nonempty budget")
        for collection in ("candidates", "metrics"):
            items = group.get(collection)
            if not isinstance(items, list) or not items:
                raise ProtocolError(f"group.{collection} must be a nonempty list")
            ids = set()
            for item in items:
                if not isinstance(item, dict) or not _text(item.get("id")) or item["id"] in ids:
                    raise ProtocolError(f"{collection} ids must be nonempty and unique within a group")
                ids.add(item["id"])
                if collection == "metrics":
                    _validate_metric(item)
                    if item.get("direction") not in ("min", "max") or not _text(item.get("unit")):
                        raise ProtocolError("each metric requires direction=min|max and a unit")
                    if "requires_fidelity_gate" in item and not isinstance(item["requires_fidelity_gate"], bool):
                        raise ProtocolError("requires_fidelity_gate must be a boolean")
        if not _same(group, group):
            raise ProtocolError("protocol contains invalid JSON values")


def _needs_gate(group: dict, metric: dict) -> bool:
    temporal = re.search(r"(^|[^a-z])(acc(?:eleration)?|jerk|smooth(?:ness)?)([^a-z]|$)", metric["id"].lower())
    return metric.get("requires_fidelity_gate", False) or (group["module"] == 4 and bool(temporal))


def _gate_reasons(run: dict, group: dict, episodes: list[dict]) -> list[str]:
    gate = run.get("fidelity_gate")
    if not isinstance(gate, dict):
        return ["fidelity_gate_missing"]
    reasons = []
    if gate.get("baseline_fingerprint") != group["reference_fingerprint"]:
        reasons.append("fidelity_gate_baseline_mismatch")
    if gate.get("review_status") != "passed" or not _text(gate.get("reviewer")):
        reasons.append("fidelity_gate_human_review_not_passed")
    evidence = gate.get("evidence")
    if not isinstance(evidence, list):
        return reasons + ["fidelity_gate_video_evidence_missing"]
    for expected in episodes:
        matches = [e for e in evidence if isinstance(e, dict) and _same(e.get("episode_id"), expected["id"])]
        if len(matches) != 1:
            reasons.append(f"episode_{expected['id']}:fidelity_evidence_missing_or_duplicate")
            continue
        item = matches[0]
        if item.get("independent_video_evidence") is not True or not _text(item.get("artifact")):
            reasons.append(f"episode_{expected['id']}:independent_video_evidence_missing")
        if not _sha(item.get("video_sha256")) or item["video_sha256"].lower() != expected["video_sha256"].lower():
            reasons.append(f"episode_{expected['id']}:fidelity_video_hash_mismatch")
    return reasons


def _evaluate(run: dict, group: dict, metric: dict, expected_episodes: list[dict]) -> dict:
    reasons = []
    values = {}
    provenance = run.get("provenance")
    if not isinstance(provenance, dict):
        reasons.append("provenance_missing")
    else:
        if provenance.get("dataset") != "track1":
            reasons.append("provenance_not_track1")
        if provenance.get("output_kind") != "reconstruction" or provenance.get("contains_fake_outputs") is not False:
            reasons.append("real_reconstruction_not_declared")
    # A copied real-provenance declaration cannot override an explicit fake marker.
    if _non_reconstruction_claim(run):
        reasons.append("non_reconstruction_output")
    if _official_claim(run):
        reasons.append("official_score_claim_not_supported")
    if run.get("status") != "complete":
        reasons.append("run_not_complete:" + str(run.get("status", "missing")))
    for field in ("evaluator", "upstream_fingerprint", "reference_fingerprint", "budget"):
        if field not in run or not _same(run[field], group[field]):
            reasons.append(field + "_mismatch")
    episodes = run.get("episodes")
    if not isinstance(episodes, list):
        episodes = []
        reasons.append("episodes_missing")
    if len(episodes) != 3 or any(not isinstance(e, dict) or not _integer(e.get("id")) or e["id"] not in (0, 1, 2) for e in episodes):
        reasons.append("episode_set_mismatch")
    for expected in expected_episodes:
        episode_id = expected["id"]
        prefix = f"episode_{episode_id}:"
        matches = [e for e in episodes if isinstance(e, dict) and _same(e.get("id"), episode_id)]
        if len(matches) != 1:
            reasons.append(prefix + "missing_or_duplicate")
            continue
        actual = matches[0]
        if "status" in actual and actual["status"] != "complete":
            reasons.append(prefix + "episode_not_complete:" + str(actual["status"]))
        if not _same(actual.get("frames"), expected["frames"]):
            reasons.append(prefix + "frame_count_mismatch")
        if not _sha(actual.get("video_sha256")) or actual["video_sha256"].lower() != expected["video_sha256"].lower():
            reasons.append(prefix + "video_hash_mismatch")
        measurements = actual.get("metrics")
        value = measurements.get(metric["id"]) if isinstance(measurements, dict) else None
        if not _number(value):
            reasons.append(prefix + "metric_missing_or_nonfinite:" + metric["id"])
        else:
            values[str(episode_id)] = value
    if _needs_gate(group, metric):
        reasons.extend(_gate_reasons(run, group, expected_episodes))
    macro = None
    if not reasons:
        try:
            macro = math.fsum(value / 3 for value in values.values())
        except OverflowError:
            pass
        if macro is None or not math.isfinite(macro):
            reasons.append("macro_mean_nonfinite")
            macro = None
    return {"status": "unranked" if reasons else "ranked", "rank": None,
            "macro_mean": macro, "episode_values": values, "reasons": reasons}


def rank_results(protocol: dict, results: dict) -> dict:
    """Validate comparison metadata and return one independent table per metric."""
    _validate_protocol(protocol)
    if not isinstance(results, dict) or not _same(results.get("schema_version"), 1) or not isinstance(results.get("runs"), list):
        raise ProtocolError("results requires schema_version=1 and a runs list")
    if "official_score" in results and results["official_score"] is not False:
        raise ProtocolError("results cannot claim official scores")
    known = {(g["id"], c["id"]) for g in protocol["groups"] for c in g["candidates"]}
    indexed = {key: [] for key in known}
    unrecognized = []
    for index, run in enumerate(results["runs"]):
        if not isinstance(run, dict) or not _text(run.get("group_id")) or not _text(run.get("candidate_id")):
            unrecognized.append({"run_index": index, "status": "unranked", "reasons": ["unknown_or_invalid_candidate"]})
            continue
        key = (run["group_id"], run["candidate_id"])
        if key not in known:
            unrecognized.append({"run_index": index, "group_id": key[0], "candidate_id": key[1],
                                 "status": "unranked", "reasons": ["unknown_or_invalid_candidate"]})
        else:
            indexed[key].append(run)
    report = {"schema_version": 1, "kind": "internal_track1_diagnostic_ranking", "official_score": False,
              "protocol_frozen": protocol["frozen"],
              "aggregation": "equal_weight_episode_macro_mean", "episode_ids": [0, 1, 2],
              "metric_scope": "prediction_and_video_diagnostics_only",
              "provenance_declarations_independently_verified": False,
              "measurement_claims_independently_verified": False, "groups": [], "unrecognized_runs": unrecognized}
    for group in protocol["groups"]:
        group_report = {"id": group["id"], "module": group["module"], "evaluator": group["evaluator"],
                        "upstream_fingerprint": group["upstream_fingerprint"],
                        "reference_fingerprint": group["reference_fingerprint"], "budget": group["budget"], "metrics": []}
        for metric in group["metrics"]:
            entries = []
            for candidate in group["candidates"]:
                runs = indexed[(group["id"], candidate["id"])]
                if len(runs) == 1:
                    entry = _evaluate(runs[0], group, metric, protocol["episodes"])
                else:
                    entry = {"status": "unranked", "rank": None, "macro_mean": None, "episode_values": {},
                             "reasons": ["no_result" if not runs else "duplicate_runs"]}
                if not protocol["frozen"]:
                    entry.update(status="unranked", rank=None, macro_mean=None)
                    entry["reasons"].insert(0, "protocol_not_frozen")
                entries.append({"candidate_id": candidate["id"], **entry})
            ranked = [entry for entry in entries if entry["status"] == "ranked"]
            ranked.sort(key=lambda e: (e["macro_mean"] if metric["direction"] == "min" else -e["macro_mean"], e["candidate_id"]))
            previous = None
            for position, entry in enumerate(ranked, 1):
                entry["rank"] = previous["rank"] if previous is not None and entry["macro_mean"] == previous["macro_mean"] else position
                previous = entry
            group_report["metrics"].append({**metric, "fidelity_gate_required": _needs_gate(group, metric),
                                            "entries": ranked + [e for e in entries if e["status"] == "unranked"]})
        report["groups"].append(group_report)
    return report


def render_markdown(report: dict) -> str:
    def cell(value: Any) -> str:
        return html.escape(str(value)).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = ["# First-three-video diagnostic rankings", "",
             "Internal Track 1 diagnostics. This tool does not compute official competition scores.", "",
             "Prediction-only self_acceleration is not official reference-relative ACC. Metrics are never combined into a total.", "",
             f"Protocol frozen: {str(report['protocol_frozen']).lower()}.", "",
             "Each metric uses the equal-weight mean of episodes 0, 1 and 2. Missing or incompatible results remain unranked.",
             "Provenance, measurement and human-review declarations are not independently verified by this metadata tool.", ""]
    for group in report["groups"]:
        for metric in group["metrics"]:
            lines += [f"## {cell(group['id'])}: {cell(metric['id'])}", "",
                      f"Unit: {cell(metric['unit'])}; direction: {cell(metric['direction'])}; fidelity gate: {metric['fidelity_gate_required']}.", "",
                      "| Rank | Candidate | Macro mean | Status | Reasons |", "| ---: | --- | ---: | --- | --- |"]
            for entry in metric["entries"]:
                mean = "—" if entry["macro_mean"] is None else format(entry["macro_mean"], ".12g")
                lines.append(f"| {entry['rank'] or '—'} | {cell(entry['candidate_id'])} | {mean} | {entry['status']} | {cell('; '.join(entry['reasons']))} |")
            lines.append("")
    if report["unrecognized_runs"]:
        lines += ["## Unrecognized results", ""]
        lines += [f"- Run {entry['run_index']}: unranked; unknown or invalid candidate." for entry in report["unrecognized_runs"]]
        lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    def invalid_constant(value: str) -> None:
        raise ProtocolError(f"nonstandard JSON constant: {value}")

    def unique_keys(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid_constant, object_pairs_hook=unique_keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = [p.resolve() for p in (args.protocol, args.results, args.output_json, args.output_md)]
        if paths[2] in paths[:2] or paths[3] in paths[:3]:
            raise ProtocolError("output paths must be distinct and must not overwrite inputs")
        report = rank_results(_read_json(args.protocol), _read_json(args.results))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        args.output_md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
