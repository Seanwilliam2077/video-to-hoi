"""Synthetic command/annotation contracts only; no videos or model weights."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def wrapper():
    path = Path(__file__).resolve().parents[1] / "tools/refine_contact_ab.py"
    spec = importlib.util.spec_from_file_location("contact_ab_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(wrapper, tmp_path):
    values = {name: str(tmp_path / name) for name in wrapper.VALUE_OPTIONS}
    values.update(wrapper.REQUIRED_VALUES)
    command = ["python", "-B", str(tmp_path / "old/tools/refine_human_hands.py")]
    for name in wrapper.VALUE_OPTIONS:
        command += ["--" + name, values[name]]
    command += ["--optimize-wrist-chain"]
    return {"episode": 16, "command": command}


def phases():
    return {
        "track": 1, "episode": 16, "frame_count": 360, "video_sha256": "synthetic-source",
        "annotation_method": "Independent synthetic fixture, not a video annotation",
        "source_review_receipt": "synthetic-review.json", "source_review_receipt_sha256": "fixture",
        "phase_intervals_inclusive": [
            {"first": 0, "last": 74, "anchor_weight": 0, "state": "unresolved"},
            {"first": 75, "last": 250, "anchor_weight": 1, "eligible_sides": ["left", "right"],
             "constraint_type": "grasp", "state": "active"},
            {"first": 251, "last": 359, "anchor_weight": 0, "state": "unresolved"},
        ],
    }


def test_pair_changes_only_output_and_existing_contact_terms(wrapper, tmp_path):
    commands, values = wrapper.paired_commands(reference(wrapper, tmp_path), tmp_path / "snapshot",
                                               tmp_path / "object.npz", tmp_path / "phases.json",
                                               tmp_path / "A", tmp_path / "B")
    a, b = commands["A"].copy(), commands["B"].copy()
    a[a.index("--output") + 1] = "OUTPUT"
    b[b.index("--output") + 1] = "OUTPUT"
    contact = b[b.index("--contact-object"):]
    assert b[:b.index("--contact-object")] == a
    assert "--contact-object" not in a and "--motion-phases" not in b
    assert contact[contact.index("--contact-scale") + 1] == "1.0"
    assert contact[contact.index("--contact-scale-mode") + 1] == "fixed_pose_geometry"
    assert values["iterations"] == "600" and values["unobserved-policy"] == "preserve"


@pytest.mark.parametrize("extra", [
    ["--contact-scale", "0.8"], ["--skip-mesh-export"], ["--motion-phases", "unexpected.json"],
    ["--contact-object", "unexpected.npz"],
])
def test_reference_cannot_leak_a_scale_or_extra_intervention(wrapper, tmp_path, extra):
    value = reference(wrapper, tmp_path)
    value["command"] += extra
    with pytest.raises(ValueError, match="Unexpected reference option"):
        wrapper.parse_reference_command(value)


@pytest.mark.parametrize("flag,value", [("--threads", "2"), ("--holdout-stride", "0"),
                                         ("--iterations", "1200"), ("--learning-rate", "0.01")])
def test_shared_frozen_budget_and_image_partition_cannot_change(wrapper, tmp_path, flag, value):
    record = reference(wrapper, tmp_path)
    record["command"][record["command"].index(flag) + 1] = value
    with pytest.raises(ValueError, match="Controlled configuration changed"):
        wrapper.parse_reference_command(record)


def test_attempts_must_be_distinct_and_preserved(wrapper, tmp_path):
    record = reference(wrapper, tmp_path)
    with pytest.raises(ValueError, match="different fresh"):
        wrapper.paired_commands(record, tmp_path, "object", "phases", tmp_path / "same", tmp_path / "same")
    (tmp_path / "A").mkdir()
    with pytest.raises(ValueError, match="different fresh"):
        wrapper.paired_commands(record, tmp_path, "object", "phases", tmp_path / "A", tmp_path / "B")


def test_release_unknown_phases_have_exactly_zero_direct_attraction(wrapper):
    weights = wrapper.validate_manual_phases(phases(), "synthetic-source")
    assert weights.shape == (360, 2)
    assert not weights[:75].any() and not weights[251:].any()
    assert np.all(weights[75:251] > 0)
    assert np.all(weights[75] < weights[90])


@pytest.mark.parametrize("kind", ["sliding", "no_contact", "release", "unresolved"])
def test_fixed_no_slip_attraction_requires_explicit_grasp(wrapper, kind):
    value = phases()
    value["phase_intervals_inclusive"][1]["constraint_type"] = kind
    with pytest.raises(ValueError, match="Only reviewed grasp"):
        wrapper.validate_manual_phases(value, "synthetic-source")


def test_phase_source_and_review_provenance_are_required(wrapper):
    with pytest.raises(ValueError, match="exact complete EP16"):
        wrapper.validate_manual_phases(phases(), "different-video")
    value = phases()
    del value["source_review_receipt_sha256"]
    with pytest.raises(ValueError, match="pinned source-review"):
        wrapper.validate_manual_phases(value, "synthetic-source")


def test_execution_rechecks_that_only_contact_is_the_intervention(wrapper, tmp_path):
    record = reference(wrapper, tmp_path)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(record))
    commands, _ = wrapper.paired_commands(record, tmp_path / "source", "object", "phases",
                                          tmp_path / "A", tmp_path / "B")
    plan = {"schema_version": wrapper.SCHEMA, "track": 1, "episode": 16, "frames": 360,
            "all30_execution_authorized": False, "object_scale_factor": 1.0, "quality_accepted": False,
            "wrapper_source_sha256": wrapper.digest(wrapper.__file__), "reference_fit_job": str(path),
            "source_root": str(tmp_path / "source"), "prepared_object": "object", "manual_phases": "phases",
            "outputs": {"A": str(tmp_path / "A"), "B": str(tmp_path / "B")},
            "commands": commands, "input_bindings": [], "fitter_source_bindings": []}
    wrapper.verify_plan(plan)
    # Output presence after a completed fit does not invalidate immutable command checks.
    (tmp_path / "A").mkdir()
    wrapper.verify_plan(plan)
    plan["commands"]["B"][plan["commands"]["B"].index("--learning-rate") + 1] = "0.02"
    with pytest.raises(ValueError, match="contact-only intervention"):
        wrapper.verify_plan(plan)


def test_execution_rejects_changed_frozen_input_bytes(wrapper, tmp_path):
    record = reference(wrapper, tmp_path)
    source = tmp_path / "source.json"
    source.write_text("original")
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(record))
    commands, _ = wrapper.paired_commands(record, tmp_path, "object", "phases", tmp_path / "A", tmp_path / "B")
    plan = {"schema_version": wrapper.SCHEMA, "track": 1, "episode": 16, "frames": 360,
            "all30_execution_authorized": False, "object_scale_factor": 1.0, "quality_accepted": False,
            "wrapper_source_sha256": wrapper.digest(wrapper.__file__), "reference_fit_job": str(reference_path),
            "source_root": str(tmp_path), "prepared_object": "object", "manual_phases": "phases",
            "outputs": {"A": str(tmp_path / "A"), "B": str(tmp_path / "B")}, "commands": commands,
            "input_bindings": [wrapper.binding(source)], "fitter_source_bindings": []}
    source.write_text("modified")
    with pytest.raises(ValueError, match="Frozen input/source bytes changed"):
        wrapper.verify_plan(plan)


def test_anchor_reference_is_b_only_and_does_not_replace_initialization(wrapper, tmp_path):
    record = reference(wrapper, tmp_path)
    without, _ = wrapper.paired_commands(record, tmp_path, "object", "phases", tmp_path / "A", tmp_path / "B")
    with_reference, _ = wrapper.paired_commands(record, tmp_path, "object", "phases", tmp_path / "A", tmp_path / "B",
                                               anchor_reference=tmp_path / "read-only-native.npz")
    assert with_reference["A"] == without["A"]
    assert with_reference["B"][:-2] == without["B"]
    assert with_reference["B"][-2:] == ["--contact-anchor-reference", str((tmp_path / "read-only-native.npz").resolve())]
    for command in with_reference.values():
        assert command[command.index("--baseline") + 1] == record["command"][record["command"].index("--baseline") + 1]


def test_anchor_reference_bytes_are_pinned_during_execution(wrapper, tmp_path):
    record = reference(wrapper, tmp_path)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(record))
    native = tmp_path / "native.npz"
    native.write_bytes(b"independent synthetic parameter placeholder")
    native_binding = wrapper.binding(native)
    commands, _ = wrapper.paired_commands(record, tmp_path, "object", "phases", tmp_path / "A", tmp_path / "B",
                                          anchor_reference=native)
    plan = {"schema_version": wrapper.SCHEMA, "track": 1, "episode": 16, "frames": 360,
            "all30_execution_authorized": False, "object_scale_factor": 1.0, "quality_accepted": False,
            "wrapper_source_sha256": wrapper.digest(wrapper.__file__), "reference_fit_job": str(reference_path),
            "source_root": str(tmp_path), "prepared_object": "object", "manual_phases": "phases",
            "outputs": {"A": str(tmp_path / "A"), "B": str(tmp_path / "B")}, "commands": commands,
            "contact_anchor_reference": native_binding,
            "input_bindings": [native_binding], "fitter_source_bindings": []}
    wrapper.verify_plan(plan)
    native.write_bytes(b"changed synthetic reference")
    with pytest.raises(ValueError, match="Frozen input/source bytes changed"):
        wrapper.verify_plan(plan)
