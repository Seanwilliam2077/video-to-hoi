"""Independent synthetic topology and display-role boundaries; no challenge data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('occluded_render', Path(__file__).resolve().parents[1] / 'tools/render_occluded_contact_ab.py')
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


def test_skin_selection_retains_original_faces_without_crossing_cut():
    faces = np.asarray([[0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 5, 6]])
    out = render.selected_hand_faces(faces, np.asarray([0, 1, 2, 4, 5, 6]), vertex_count=7)
    assert np.array_equal(out, faces[[0, 3]])
    assert np.array_equal(faces, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 5, 6]])


def test_new_comparison_rejects_different_object_hypotheses():
    columns = [{'role': role, 'native': {'path': role}, 'object': {'path': obj}}
               for role, obj in [('old_a', 'old'), ('a', 'new'), ('b', 'silently-retuned')]]
    with pytest.raises(ValueError, match='exact same object'):
        render.validate_columns({'columns': columns})


def test_old_context_may_use_old_object_but_cannot_replace_new_roles():
    columns = [{'role': role, 'native': {'path': role}, 'object': {'path': obj}}
               for role, obj in [('old_a', 'old'), ('a', 'new'), ('b', 'new')]]
    assert render.validate_columns({'columns': columns}) == columns
    columns[0]['role'] = 'a'
    with pytest.raises(ValueError, match='fixed order'):
        render.validate_columns({'columns': columns})


def test_objects_must_match_evaluated_bytes_even_when_new_arms_agree(tmp_path):
    old = {'path': str(tmp_path / 'old.npz'), 'sha256': 'old'}
    new = {'path': str(tmp_path / 'new.npz'), 'sha256': 'new'}
    altered = {'path': str(tmp_path / 'retuned.npz'), 'sha256': 'retuned'}
    columns = [{'role': role, 'object': obj} for role, obj in [('old_a', old), ('a', new), ('b', new)]]
    render.validate_evaluated_objects(columns, {'old_object': old, 'object': new})
    columns[1]['object'] = altered
    columns[2]['object'] = altered
    with pytest.raises(ValueError, match='independently evaluated'):
        render.validate_evaluated_objects(columns, {'old_object': old, 'object': new})
