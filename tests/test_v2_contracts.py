"""Independent synthetic records exercise cross-file invariants, never model quality."""
import copy
import io
from uuid import uuid4

import numpy as np
import pytest

from v2hoi.v2 import IMPLEMENTATION_PROFILE, SPEC_REVISION
from v2hoi.v2.artifacts import ArtifactStore, ContractError, content_hash
from v2hoi.v2.contracts import EpisodeEvidence, SceneState, SolvePlan, ValidationContext, scene_hash


class SyntheticScene:
    """Small known-answer store with visibly synthetic provenance and trust policy."""

    def __init__(self, root):
        self.store = ArtifactStore(root)
        self.source = self.blob(b'independent synthetic pixels; no challenge input')
        self.config = self.js({'fixture': 'independently constructed five-frame scene'})
        self.validator = self.blob(b'synthetic validator identity only')
        self.trusted = {(kind, self.validator['sha256'], self.config['sha256']) for kind in ('execution', 'quality', 'export')}
        self.ctx = ValidationContext(self.store, source_kind='independent_synthetic', trusted_reports=self.trusted)
        self.provenance = {'source_kind': 'independent_synthetic', 'source_refs': [self.source], 'parent_refs': [], 'operation': 'independent_fixture', 'code_revision': 'independent-fixture-v1', 'configuration_ref': self.config, 'configuration_hash': self.config['sha256'], 'manual_edits': [], 'policy_review_ref': self.js({'decision': 'approved', 'source_kind': 'independent_synthetic', 'source_hashes': [self.source['sha256']], 'reviewer': 'synthetic-fixture-builder', 'limitations': ['no real video']}), 'producer': {'invocation': ['independent-fixture'], 'seed': 0, 'environment': {'runtime': 'synthetic'}, 'nondeterminism': []}, 'dependencies': {'config': self.config}}
        self.timeline = self.js({'frame_count': 5, 'frame_ids': list(range(5)), 'pts': list(range(5)), 'time_base': [1, 30], 'width': 64, 'height': 48, 'decode_ref': self.config})
        self.partition = self.js({'fit': ['fit'], 'review': ['review'], 'diagnostic': [], 'unused': [], 'scope': 'local_repair', 'selection_config_ref': self.config, 'selected_at': '2026-10-13T00:00:00Z', 'temporal_exclusion_frames': 1, 'limitations': ['synthetic fixture only'], 'whole_video_model_consumed_review': False})
        observations = [self.observation('fit', 0), self.observation('review', 4)]
        self.evidence = self.envelope('EpisodeEvidence', {'episode_id': 0, 'video': self.source, 'timeline': self.timeline, 'entities': [{'id': 'person', 'role': 'target_human', 'evidence_ref': self.source}, {'id': 'object', 'role': 'target_object', 'evidence_ref': self.source}], 'observations': observations, 'events': [], 'partitions': self.partition, 'evidence_policy_ref': self.config})
        self.evidence_ref = EpisodeEvidence.from_dict(self.evidence, self.ctx).publish(self.ctx).to_dict()
        self.camera = self.js({'width': 64, 'height': 48, 'K': [[50, 0, 32], [0, 50, 24], [0, 0, 1]], 'distortion': 'none', 'evidence_ref': self.source})
        self.native = self.npz(pose=np.zeros((5, 136)), shape=np.zeros(45), scales=np.zeros(68), frame_ids=np.arange(5))
        self.mesh = self.npz(vertices=np.array([[0., 0, 0], [0.2, 0, 0], [0, 0.2, 0]]), faces=np.array([[0, 1, 2]]))
        self.motion = self.npz(rotations=np.repeat(np.eye(3)[None], 5, axis=0), translations=np.array([[0., 0, 2]] * 5), frame_ids=np.arange(5))
        self.canonical = self.js({'T_new_from_source': np.eye(4).tolist()})
        self.rig = self.blob(b'independent synthetic rig marker, not MHR assets')
        self.decoder = self.blob(b'independent synthetic decoder marker')
        self.conversion = self.js({'rig_sha256': self.rig['sha256'], 'decoder_sha256': self.decoder['sha256'], 'limitations': ['no real decoder execution']})
        self.contact = self.js({'supports': [], 'edges': []})
        execution = self.js({'job_id': str(uuid4()), 'backend': 'independent_synthetic', 'source_kind': 'independent_synthetic', 'timeline_ref': self.timeline, 'inputs': [self.mesh, self.evidence_ref], 'object_inputs': [{'entity_id': 'object', 'mesh_sha256': self.mesh['sha256'], 'canonical_sha256': self.canonical['sha256'], 'scale_mode': 'baked_metric', 'scale': 1}], 'outputs': [self.native, self.motion, self.camera, self.contact], 'state': 'complete'})
        self.scene = self.envelope('SceneState', {'scene': {'episode_id': 0, 'evidence_ref': self.evidence_ref, 'timeline_ref': self.timeline, 'coordinate_frame': 'opencv_camera_m', 'camera_ref': self.camera, 'humans': [{'entity_id': 'person', 'native_ref': self.native, 'rig_ref': self.rig, 'decoder_ref': self.decoder, 'conversion_ref': self.conversion, 'native_units': 'decoder_defined'}], 'objects': [{'entity_id': 'object', 'mesh_ref': self.mesh, 'trajectory_ref': self.motion, 'canonical_transform_ref': self.canonical, 'scale_mode': 'baked_metric', 'scale': 1, 'source_episodes': [0], 'symmetry': {'type': 'none'}, 'pose_binding': {'mesh_sha256': self.mesh['sha256'], 'canonical_sha256': self.canonical['sha256'], 'scale_mode': 'baked_metric', 'scale': 1}, 'evidence_status': ['observed', 'inferred', 'unresolved', 'inferred', 'observed'], 'uncertainty_ref': self.config}], 'support_contact_ref': self.contact, 'hypothesis_lineage_ref': self.config, 'execution_manifest_ref': execution, 'preparation_refs': [self.config], 'dependencies': {'config': self.config}}, 'scene_hash': '', 'status': {'execution_valid': False, 'quality_accepted': False, 'export_valid': False, 'selection': 'candidate'}, 'validation_reports': []})
        self.rehash(self.scene)

    def blob(self, data):
        return self.store.put_bytes(data, 'application/octet-stream').to_dict()

    def js(self, data):
        return self.store.put_json(data).to_dict()

    def npz(self, **values):
        out = io.BytesIO()
        np.savez(out, **values)
        return self.store.put_bytes(out.getvalue(), 'application/x-npz', suffix='.npz').to_dict()

    def envelope(self, kind, payload):
        return {'record_type': kind, 'spec_revision': SPEC_REVISION, 'implementation_profile': IMPLEMENTATION_PROFILE, 'id': str(uuid4()), 'created_at': '2026-10-13T00:00:00Z', 'provenance': copy.deepcopy(self.provenance), 'payload': payload}

    def observation(self, identity, frame):
        return {'observation_id': identity, 'entity_id': 'object', 'frame_ids': [frame], 'kind': 'landmarks', 'origin': 'human_annotation', 'data_ref': self.js({'synthetic_frame': frame, 'xy': [2, 3]}), 'producer': {'name': 'independent fixture'}, 'coordinate_frame': 'source_pixels', 'visibility': 'visible', 'validity_ref': self.config, 'derivation_refs': [], 'derived_from': [], 'lineage_roots': ['synthetic-pixels-' + str(frame)], 'image_transform': {'source_size': [64, 48], 'processed_size': [64, 48], 'pixel_centres': 'integer_pixel_centres', 'T_processed_from_source': np.eye(3).tolist()}}

    def rehash(self, record):
        record['payload']['scene_hash'] = scene_hash(record)

    def attest(self, record, kinds=('execution',)):
        previous = SceneState.from_dict(record, self.ctx).publish(self.ctx)
        record = copy.deepcopy(record)
        record['id'] = str(uuid4())
        status = record['payload']['status']
        reports = []
        for kind, flag in [('execution', 'execution_valid'), ('quality', 'quality_accepted'), ('export', 'export_valid')]:
            status[flag] = kind in kinds
            if kind in kinds:
                reports.append(self.js({'kind': kind, 'scene_hash': record['payload']['scene_hash'], 'passed': True, 'validator_ref': self.validator, 'configuration_ref': self.config, 'checks': [{'name': 'synthetic fixture assertion', 'passed': True}], 'limitations': ['no actual reconstruction or official packing']}))
        status['selection'] = 'accepted' if 'quality' in kinds else 'candidate'
        record['payload']['validation_reports'] = reports
        record['payload']['review_lineage'] = {'previous_ref': previous.to_dict(), 'operation': 'synthetic review', 'report_refs': reports}
        return record

    def plan(self):
        parent = SceneState.from_dict(self.attest(self.scene), self.ctx).publish(self.ctx).to_dict()
        acceptance = {'frozen': True, 'metrics': ['synthetic residual'], 'tolerances': {'synthetic': 0}, 'allowed_regressions': {}, 'review_criteria': ['synthetic only'], 'full_clip': True, 'boundary_checks': True, 'required_reports': ['quality'], 'limitations': ['no real thresholds']}
        return self.envelope('SolvePlan', {'input_scene_ref': parent, 'evidence_ref': self.evidence_ref, 'operation': 'orientation_branch', 'entity_ids': ['object'], 'intervals': [{'start_frame_id': 1, 'end_frame_id': 3}], 'mutable_variables': [{'entity_id': 'object', 'block': 'rotation', 'frame_ids': [1, 2, 3]}], 'fixed_variables': ['camera', 'mesh', 'scale', 'human_shape', 'human_scales', 'all_undeclared', 'human'], 'bounds': {'max_rotation_change_rad': 0.2}, 'fit_partition_ref': self.partition, 'review_partition_ref': self.partition, 'budgets': {'max_candidates': 2, 'max_iterations': 10, 'max_wall_seconds': 10, 'max_gpu_memory_bytes': 1024}, 'acceptance_config_ref': self.js(acceptance), 'acceptance_config_hash': content_hash(acceptance), 'rollback_scene_ref': parent, 'failure_action': 'retain_parent', 'unresolved_action': 'retain_parent_and_record_uncertainty', 'operator_ref': self.config})


@pytest.fixture
def fixture(tmp_path):
    return SyntheticScene(tmp_path / 'artifacts')


def test_round_trip_records_and_detached_views(fixture):
    f = fixture
    record = SceneState.from_dict(f.scene, f.ctx)
    mutated = record.to_dict()
    mutated['payload']['scene']['episode_id'] = 29
    assert record.to_dict()['payload']['scene']['episode_id'] == 0
    ref = record.publish(f.ctx)
    assert SceneState.load(ref, f.ctx).to_dict() == f.scene
    assert SolvePlan.from_dict(f.plan(), f.ctx).publish(f.ctx)


@pytest.mark.parametrize('kind', ['track2', 'unknown', 'track1'])
def test_source_policy_is_checked_before_payload_open(fixture, kind):
    record = copy.deepcopy(fixture.evidence)
    record['provenance']['source_kind'] = kind
    record['payload']['video']['path'] = 'never-open-this'
    with pytest.raises(ContractError, match='source domain'):
        fixture.ctx.validate(record)


@pytest.mark.parametrize('mutation', ['duplicate_bytes', 'derived_from', 'derivation_ref', 'root'])
def test_fit_review_lineage_leakage(fixture, mutation):
    record = copy.deepcopy(fixture.evidence)
    fit, review = record['payload']['observations']
    if mutation == 'duplicate_bytes':
        review['data_ref'] = fit['data_ref']
    elif mutation == 'derived_from':
        review['derived_from'] = [fit['observation_id']]
    elif mutation == 'derivation_ref':
        review['derivation_refs'] = [fit['data_ref']]
    else:
        review['lineage_roots'] = fit['lineage_roots']
    with pytest.raises(ContractError, match='lineage overlaps'):
        fixture.ctx.validate(record)


def test_temporal_buffer_and_crop_mapping(fixture):
    record = copy.deepcopy(fixture.evidence)
    record['payload']['observations'][1]['frame_ids'] = [1]
    with pytest.raises(ContractError, match='temporal exclusion'):
        fixture.ctx.validate(record)


def test_declared_producer_config_must_match_its_bytes(fixture):
    record = copy.deepcopy(fixture.evidence)
    record['provenance']['configuration_hash'] = 'a' * 64
    with pytest.raises(ContractError, match='producer configuration hash'):
        fixture.ctx.validate(record)
    record = copy.deepcopy(fixture.evidence)
    record['payload']['observations'][0]['image_transform']['source_size'] = [32, 24]
    with pytest.raises(ContractError, match='source image'):
        fixture.ctx.validate(record)


@pytest.mark.parametrize('mutation,message', [('mesh', 'pose binding'), ('rig', 'rig/decoder'), ('missing_frame', 'frame'), ('double_scale', 'scale'), ('legacy_human', 'pose'), ('mixed_job', 'coupled output'), ('cross_episode', 'pooling')])
def test_scene_binding_and_array_failures(fixture, mutation, message):
    f = fixture
    record = copy.deepcopy(f.scene)
    scene = record['payload']['scene']
    obj = scene['objects'][0]
    if mutation == 'mesh':
        obj['pose_binding']['mesh_sha256'] = 'a' * 64
    elif mutation == 'rig':
        scene['humans'][0]['rig_ref'] = f.decoder
    elif mutation == 'missing_frame':
        obj['trajectory_ref'] = f.npz(rotations=np.repeat(np.eye(3)[None], 4, axis=0), translations=np.zeros((4, 3)), frame_ids=np.arange(4))
    elif mutation == 'double_scale':
        obj['scale'] = 2
        obj['pose_binding']['scale'] = 2
    elif mutation == 'legacy_human':
        scene['humans'][0]['native_ref'] = f.npz(pose=np.zeros((5, 77, 3)), scales=np.zeros(68), shape=np.zeros(45), frame_ids=np.arange(5))
    elif mutation == 'mixed_job':
        invocation = f.ctx.json(scene['execution_manifest_ref'])
        invocation['outputs'] = [f.native]
        scene['execution_manifest_ref'] = f.js(invocation)
    else:
        obj['source_episodes'] = [0, 1]
    f.rehash(record)
    with pytest.raises(ContractError, match=message):
        f.ctx.validate(record)


def test_nested_dependency_tamper_is_detected(fixture):
    path = fixture.store.root / fixture.validator['path']
    record = fixture.attest(fixture.scene)
    path.write_bytes(b'tampered validator')
    with pytest.raises(ContractError, match='hash'):
        fixture.ctx.validate(record)


def test_supplemental_dependency_manifest_is_transitively_verified(fixture):
    f = fixture
    leaf = f.blob(b'independent synthetic supplemental dependency')
    record = copy.deepcopy(f.scene)
    record['payload']['scene']['dependencies']['supplemental'] = f.js({'nested': leaf})
    f.rehash(record)
    f.ctx.validate(record)
    (f.store.root / leaf['path']).write_bytes(b'changed dependency')
    with pytest.raises(ContractError, match='hash'):
        f.ctx.validate(record)


def test_atomic_invocation_requires_matching_evidence(fixture):
    f = fixture
    record = copy.deepcopy(f.scene)
    execution = f.ctx.json(record['payload']['scene']['execution_manifest_ref'])
    execution['inputs'] = [f.mesh]
    record['payload']['scene']['execution_manifest_ref'] = f.js(execution)
    f.rehash(record)
    with pytest.raises(ContractError, match='EpisodeEvidence input'):
        f.ctx.validate(record)


def test_invocation_cannot_be_reinterpreted_with_a_new_scale(fixture):
    f = fixture
    record = copy.deepcopy(f.scene)
    obj = record['payload']['scene']['objects'][0]
    obj.update(scale_mode='explicit_uniform', scale=2)
    obj['pose_binding'].update(scale_mode='explicit_uniform', scale=2)
    f.rehash(record)
    with pytest.raises(ContractError, match='scale/canonical bindings'):
        f.ctx.validate(record)


@pytest.mark.parametrize('mutation', ['operation', 'reports', 'missing'])
def test_status_review_must_link_decision_and_previous_scene(fixture, mutation):
    f = fixture
    record = f.attest(f.scene)
    if mutation == 'operation':
        record['payload']['review_lineage']['operation'] = ''
    elif mutation == 'reports':
        record['payload']['review_lineage']['report_refs'] = []
    else:
        del record['payload']['review_lineage']
    with pytest.raises(ContractError, match='review|report refs'):
        f.ctx.validate(record)


def test_status_flags_require_trusted_current_reports(fixture):
    f = fixture
    record = copy.deepcopy(f.scene)
    record['payload']['status']['execution_valid'] = True
    with pytest.raises(ContractError, match='passing report'):
        f.ctx.validate(record)
    attested = f.attest(f.scene)
    untrusted = ValidationContext(f.store, source_kind='independent_synthetic')
    with pytest.raises(ContractError, match='untrusted'):
        untrusted.validate(attested)
    attested['payload']['scene']['hypothesis_lineage_ref'] = f.source
    f.rehash(attested)
    with pytest.raises(ContractError, match='stale validation report'):
        f.ctx.validate(attested)


def test_status_revision_preserves_scene_identity(fixture):
    f = fixture
    original = SceneState.from_dict(f.scene, f.ctx).publish(f.ctx)
    reviewed = f.attest(f.scene, ('execution', 'quality'))
    reviewed['id'] = str(uuid4())
    reviewed['payload']['review_lineage'] = {'previous_ref': original.to_dict(), 'operation': 'independent synthetic review', 'report_refs': reviewed['payload']['validation_reports']}
    new_ref = SceneState.from_dict(reviewed, f.ctx).publish(f.ctx)
    assert reviewed['payload']['scene_hash'] == f.scene['payload']['scene_hash']
    assert f.ctx.select_scene('accepted', new_ref) == new_ref
    with pytest.raises(ContractError, match='export validity'):
        f.ctx.select_scene('release', new_ref, require_export=True)
    assert f.store.get_selection('release') is None


def test_invalid_fallback_cannot_replace_accepted_selection(fixture):
    f = fixture
    accepted = f.attest(f.scene, ('execution', 'quality', 'export'))
    ref = SceneState.from_dict(accepted, f.ctx).publish(f.ctx)
    f.ctx.select_scene('accepted', ref, require_export=True)
    invalid = copy.deepcopy(f.scene)
    invalid['id'] = str(uuid4())
    invalid_ref = SceneState.from_dict(invalid, f.ctx).publish(f.ctx)
    with pytest.raises(ContractError, match='unaccepted'):
        f.ctx.select_scene('accepted', invalid_ref, expected=ref)
    assert f.store.get_selection('accepted') == ref


@pytest.mark.parametrize('mutation,message', [('global', 'operation'), ('human', 'variable block'), ('outside', 'intervals'), ('fixed_overlap', 'overlap'), ('translation_only', 'rotation block'), ('budget', 'integer'), ('missing_bound', 'explicit bound'), ('unfrozen', 'frozen'), ('policy_hash', 'configuration hash')])
def test_bounded_plan_rejects_scope_and_budget_changes(fixture, mutation, message):
    f = fixture
    plan = f.plan()
    p = plan['payload']
    if mutation == 'global':
        p['operation'] = 'scale_repair'
    elif mutation == 'human':
        p['entity_ids'].append('person')
        p['mutable_variables'][0].update(entity_id='person', block='hand')
    elif mutation == 'outside':
        p['mutable_variables'][0]['frame_ids'] = [0, 1]
    elif mutation == 'fixed_overlap':
        p['fixed_variables'].append('object.rotation')
    elif mutation == 'translation_only':
        p['mutable_variables'][0]['block'] = 'translation'
    elif mutation == 'budget':
        p['budgets']['max_candidates'] = 0
    elif mutation == 'missing_bound':
        p['bounds'] = {'some_other_bound': 1}
    elif mutation == 'policy_hash':
        p['acceptance_config_hash'] = 'a' * 64
    else:
        config = f.ctx.json(p['acceptance_config_ref'])
        config['frozen'] = False
        p['acceptance_config_ref'] = f.js(config)
        p['acceptance_config_hash'] = content_hash(config)
    with pytest.raises(ContractError, match=message):
        f.ctx.validate(plan)


def test_sliding_contact_cannot_be_rigid_grasp(fixture):
    f = fixture
    record = copy.deepcopy(f.scene)
    edge = {'id': 'wipe', 'entity_ids': ['person', 'object'], 'type': 'sliding', 'state': 'candidate', 'interval': {'start_frame_id': 0, 'end_frame_id': 4}, 'evidence_refs': [f.source], 'patches': ['palm', 'surface'], 'normal_freedom': 'soft', 'tangential_freedom': 'constrained', 'weight_policy_ref': f.config}
    record['payload']['scene']['support_contact_ref'] = f.js({'supports': [], 'edges': [edge]})
    f.rehash(record)
    with pytest.raises(ContractError, match='no-slip'):
        f.ctx.validate(record)


def test_passing_attestation_cannot_hide_a_failed_check(fixture):
    f = fixture
    record = f.attest(f.scene)
    report = f.ctx.json(record['payload']['validation_reports'][0])
    report['checks'][0]['passed'] = False
    refs = [f.js(report)]
    record['payload']['validation_reports'] = refs
    record['payload']['review_lineage']['report_refs'] = refs
    with pytest.raises(ContractError, match='failed or unrun'):
        f.ctx.validate(record)


def test_contact_human_edit_requires_matching_decoder_control_map(fixture):
    f = fixture
    plan = f.plan()
    p = plan['payload']
    p.update(operation='contact_transition', entity_ids=['person'], bounds={'max_native_control_change': 0.1})
    p['fixed_variables'].remove('human')
    control = {'decoder_sha256': f.decoder['sha256'], 'rig_sha256': f.rig['sha256'], 'block': 'hand', 'indices': [120], 'units': 'independent synthetic unit', 'review_ref': f.config}
    p['mutable_variables'] = [{'entity_id': 'person', 'block': 'hand', 'frame_ids': [1, 2, 3], 'control_map_ref': f.js(control)}]
    f.ctx.validate(plan)
    control['decoder_sha256'] = 'a' * 64
    p['mutable_variables'][0]['control_map_ref'] = f.js(control)
    with pytest.raises(ContractError, match='control map binding'):
        f.ctx.validate(plan)


@pytest.mark.parametrize('kind', ['canonical', 'symmetry'])
def test_improper_coordinate_and_symmetry_metadata(fixture, kind):
    f = fixture
    record = copy.deepcopy(f.scene)
    obj = record['payload']['scene']['objects'][0]
    if kind == 'canonical':
        matrix = np.diag([-1., 1, 1, 1])
        obj['canonical_transform_ref'] = f.js({'T_new_from_source': matrix.tolist()})
        obj['pose_binding']['canonical_sha256'] = obj['canonical_transform_ref']['sha256']
    else:
        obj['symmetry'] = {'type': 'continuous', 'axis': [0, 0, 0], 'origin': [0, 0, 0], 'evidence_ref': f.source}
    f.rehash(record)
    with pytest.raises(ContractError, match='proper rigid|unit axis'):
        f.ctx.validate(record)
