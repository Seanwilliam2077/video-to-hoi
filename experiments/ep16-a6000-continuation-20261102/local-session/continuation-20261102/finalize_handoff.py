from pathlib import Path
import json,datetime,hashlib
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');E=R/'experiments/ep16-joint-direction-20261031';T=R/'mast3r-pilot';C=T/'source-interior-correction-v1';P=T/'object-cue-13frames-v1'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
correspondence={'status':'NOT_QUALIFIED_FOR_FULL_WINDOW_NATIVE_POSE_UPDATE','reviewed_source_pairs':[[150,155],[170,175],[205,210]],'published_mask_semantics':'Rejected: stationary table, despite correct source/video/frame hashes.','corrected_source_support':'13 source-only positive visible-box interior polygons, reviewed once; no full silhouette or material identity truth implied.','corrected_candidates_per_pair':[60,88],'fixed_roundtrip_tolerance_original_pixels':1.7578125,'first_two_forward_roundtrip_counts':[2,6],'full_13_frame_tracks':0,'limits':['This is a result for this fixed model, crop, matcher, support and association rule; not a proof that monocular rotation recovery is impossible.','Coarse sampling, repeated texture, viewpoint changes and occlusion can break or alias correspondences.','Spatial coverage and roundtrip consistency do not certify physical material-point identity.'],'decision':'Retain raw predictions and corrected candidates. Do not fit or promote a new full-window native pose from this evidence; do not tune thresholds to force qualification.','native_pose_update_started':False,'additional_model_forwards_after_original_24':0}
(C/'qualification-decision.json').write_text(json.dumps(correspondence,indent=2))
selection={'published_object_mask_package':'input-evidence-20261022-ep16-object-masks','published_masks_status_for_object_cue':'REJECTED_SEMANTIC_MISMATCH_STATIONARY_TABLE','cached_native_postopt_object_masks':'Separately audited at source frames 150/175/190/210 by K_full @ inverse(K_crop); these cover the held box. Do not conflate them with the bad published package.','new_object_support':str(C/'source-box-interiors.json'),'support_role':'Positive visible-box interior qualification only; not a replacement full timeline segmentation or official ground truth.','raw_predictions':str(P/'raw'),'trajectory_fit_enabled':False,'reason':'Fixed source/cross-direction/temporal qualification did not establish full-window motion constraints.'}
(T/'observation-selection.json').write_text(json.dumps(selection,indent=2))
handoff={
'updated_at_utc':now,
'phase':'MIGRATION_AND_A6000_REPLAY_COMPLETE_NEW_OBJECT_CUE_NOT_QUALIFIED',
'current_root':str(R),'convenient_entry':'/data1/defu/zly/nvda/current',
'previous_source_handoff':str(R/'source-original/experiments/ep16-joint-direction-20261031/LATEST-RESEARCH-HANDOFF-20261102.json'),
'authorization':{'source':'Current user request in the Codex chat: migrate the latest code to A6000 and continue optimization there.','specified_checkpoint_and_bounded_pilot':'Executed on A6000 as the proposed next continuation step. The previous source-only pending-scope state is historical, not the current run state.','later_network_steering':'All external downloads use a proxy or a mirror; no direct upstream fallback.'},
'completed':{'migration':{'sha_verified_source_files':2507,'bytes':622894621,'path_adapted_files':412,'original_source_snapshot_preserved':True,'original_a6000_tree_preserved':True},'required_synthetic_suite':{'passed':273,'log':str(R/'a6000-runs/required-tests-clean-source.log'),'initial_runtime_tree_failure':'Asset downloader guards correctly rejected runtime weight symlinks. The explicit synthetic suite passed from the clean source copy, with no real asset loading.'},'full_native_paired_replay':json.loads((R/'a6000-runs/ep16-migrated-reference-v1/report.json').read_text()),'mast3r_pilot':json.loads((P/'report.json').read_text()),'model':json.loads((T/'model-receipt.json').read_text()),'source_mask_semantic_fix':selection,'correspondence_qualification':correspondence},
'current_paired_reference':{'sha256':'61decc2bae1b4c5db67c30c7aee521edf61e5c1f5a352e70eec1f03c57234f32','path':str(E/'joint-green-v1/refined.pth'),'accepted':False,'human_object_pair_preserved':True,'native_parameters_changed_this_continuation':False},
'next':[
'Reuse the preserved code, source masks audit, 24 raw forwards and fixed correspondence qualification; do not download the checkpoint or repeat this pilot.',
'Object rotation optimization remains unexecuted in this continuation because the new observation route did not qualify for the full window. The new mask semantic correction is for cue selection, not an accepted trajectory correction.',
'Any further estimator should explicitly address weak initial edges and repeated-pattern/material identity before a bounded pose candidate. Preserve frame 150/210 tilt and frame 300 release, original source frame IDs, human-object pairing, shared camera and units.',
'Keep Track 1-only restrictions. Do not restart closed depth/scale/angle/controller scans or claim true contact, official PEN/ACC, accepted physical scale or submission success.',
'External downloads must use the existing local proxy or an explicit mirror, with pinned revisions and original SHA checks. The successful checkpoint transfer used hf-mirror.com and the original expected SHA.',
'The user requested continued optimization. Report honestly that migration and the bounded observation experiment completed, while new native pose optimization has not yet started; there is no background GPU optimizer waiting.'
],
'gpu_jobs_pending':[],
'important_outputs':{'full_replay':str(R/'a6000-runs/ep16-migrated-reference-v1/ep16-source-and-native-360-30fps.mp4'),'bad_mask_evidence':str(P/'source-mask-review.jpg'),'corrected_source_regions':str(C/'source-region-review.jpg'),'temporal_association':str(C/'temporal-association-report.json'),'qualification_decision':str(C/'qualification-decision.json')}
}
(R/'LATEST-A6000-HANDOFF.json').write_text(json.dumps(handoff,indent=2))
# Replace only the copied runnable handoff. The original source file is untouched.
p=E/'LATEST-RESEARCH-HANDOFF-20261102.json';tmp=p.with_suffix('.a6000-tmp');tmp.write_text(json.dumps(handoff,indent=2));tmp.replace(p)
(R.parent/'ACTIVE_CONTINUATION.json').write_text(json.dumps({'root':str(R),'handoff':str(R/'LATEST-A6000-HANDOFF.json'),'phase':handoff['phase'],'updated_at_utc':now,'GPU_jobs_pending':[]},indent=2))
for p in [R/'LATEST-A6000-HANDOFF.json',C/'qualification-decision.json',T/'observation-selection.json']:json.loads(p.read_text())
print(json.dumps({'handoff':str(R/'LATEST-A6000-HANDOFF.json'),'phase':handoff['phase'],'GPU_jobs_pending':[]},indent=2))
