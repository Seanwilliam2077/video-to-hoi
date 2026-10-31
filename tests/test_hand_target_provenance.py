"""Synthetic metadata only; no videos, weights or challenge assets."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def classifier():
    source = Path(__file__).resolve().parents[1] / "tools/refine_human_hands.py"
    spec = importlib.util.spec_from_file_location("hand_target_metadata_classifier", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.target_provenance


@pytest.mark.parametrize("schema", ["egohand_geometric_proposals.v1", "refined_egohand_proposals.v1"])
def test_generated_and_hybrid_targets_never_claim_independent_mediapipe(classifier, schema):
    report = {"schema_version": schema, "shared_inputs": ["detector wrist"]}
    result = classifier(report, {"observation_kind": np.asarray("egohand_geometric_proposals.v1")})
    assert result["generated_model_proposal_provenance"] == report
    assert "independent_observation_provenance" not in result
    assert result["method"] == "source_bound_model_target_native_mhr_sequence_fit"
    assert "not independent image evidence" in result["observation_semantics"]


def test_generated_array_tag_is_sufficient_when_report_has_no_legacy_schema(classifier):
    result = classifier({"schema_version": "future_generated_schema"},
                        {"observation_kind": np.asarray("egohand_geometric_proposals.v1")})
    assert "generated_model_proposal_provenance" in result
    assert "independent_observation_provenance" not in result


def test_known_detector_remains_an_image_observer(classifier):
    report = {"observer": "MediaPipe HandLandmarker independent 2D observations"}
    result = classifier(report, {})
    assert result["independent_observation_provenance"] == report
    assert "generated_model_proposal_provenance" not in result


def test_unknown_observer_is_not_silently_called_mediapipe(classifier):
    result = classifier({"observer": "unverified fixture"}, {})
    assert "unclassified_target_provenance" in result
    assert "independent_observation_provenance" not in result


def test_nonscalar_kind_cannot_be_used_as_a_provenance_tag(classifier):
    with pytest.raises(ValueError, match="scalar"):
        classifier({}, {"observation_kind": np.asarray(["a", "b"])})
