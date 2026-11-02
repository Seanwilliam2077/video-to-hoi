"use strict";

// Independent synthetic metadata only: no challenge media or model execution.
const {test} = require("node:test");
const assert = require("node:assert/strict");
const delivery = require("../docs/progress-board/delivery-view.js");
const clone = value => JSON.parse(JSON.stringify(value));
const base = "https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-independent-object-fixture";
const asset = name => ({url: `${base}/${name}`, sha256: "a".repeat(64), bytes: 123});

function entry(stage = "objects") {
  const archive = asset("independent-object-fixture.zip"), checks = asset("producer-checks.json");
  const record = {id: `fixture-${stage}`, episode_id: "episode_000000", stage, iteration: "fixture-v1",
    state: "submitted", kind: "real_track1", created_at: "2026-11-02T12:00:00Z", contributor: "Independent fixture",
    reviewer: "", summary: "Synthetic metadata only", delivery_url: base, source_commit: "",
    coverage: {frames: 5, objects_complete: true}, upstream: [],
    artifacts: (stage === "objects" ? ["object_record", "mesh"] : ["transforms"]).map(requirement_id => ({requirement_id, label: "Fixture", ...archive})),
    checks: {structural: {status: "pass", evidence_url: checks.url}, quality: {status: "not_run", evidence_url: ""}, export: {status: "not_run", evidence_url: ""}},
    blocker: "Receiver review pending", next_action: "Inspect compatibility"};
  return {record, episode_id: record.episode_id, profile: "temporal_object_hoi_review.v1", frame_count: 5,
    object_name: "test_cube", source_video_sha256: "b".repeat(64), selected_result_sha256: "c".repeat(64), archive,
    manifest: asset("manifest.json"), checks,
    preview: {...asset("episode_000000_test_cube.mp4"), content_type: "video/mp4", original_frame_count: 5},
    quality_accepted: false, metric_scale_verified: false, official_export_accepted: false};
}
const catalog = value => ({schema_version: "hoi_object_catalog.v1", entries: [value || entry()]});

test("Objects and Motion bind exact records and render full videos in both languages", () => {
  for (const stage of ["objects", "motion"]) {
    const item = entry(stage), value = catalog(item);
    assert.deepEqual(delivery.validateObjectCatalog(value), []);
    assert.deepEqual(delivery.objectEntryFor(value, clone(item.record)), item);
    for (const language of ["en", "zh"]) {
      const html = delivery.renderObject(item, language);
      assert.match(html, /<video controls playsinline preload="none"/);
      assert.match(html, /object_scale = 1/);
      assert.match(html, /unaccepted|尚未验收/);
      assert.match(html, /confidence|置信度/);
    }
    const changed = clone(item.record); changed.iteration = "another-iteration";
    assert.equal(delivery.objectEntryFor(value, changed), null);
  }
});

test("wrong episode, release, coverage, artifact binding or acceptance cannot supply a preview", () => {
  const mutations = [
    x => x.preview.url = x.preview.url.replace("episode_000000", "episode_000001"),
    x => x.preview.url = x.preview.url.replace("result-independent", "result-other"),
    x => x.preview.original_frame_count = 4,
    x => x.record.artifacts[0].sha256 = "d".repeat(64),
    x => x.quality_accepted = true,
    x => x.record.state = "accepted",
    x => x.record.source_commit = "not-a-commit",
    x => x.record.artifacts = [null],
    x => x.preview = null,
    x => x.record = null
  ];
  for (const mutate of mutations) {
    const item = entry(); mutate(item);
    assert.ok(delivery.validateObjectCatalog(catalog(item)).length);
    assert.equal(delivery.renderObject(item), "");
  }
  assert.ok(delivery.validateObjectCatalog({schema_version: "hoi_object_catalog.v1", entries: [entry(), entry()]}).length);
});

test("only canonical repository release URLs can be embedded", () => {
  for (const url of ["http://127.0.0.1/episode_000000_test_cube.mp4", "https://github.com.evil.test/x.mp4", base+"/../episode_000000_test_cube.mp4", base+"/episode_000000_test_cube.mp4?token=secret"]) {
    const item = entry(); item.preview.url = url;
    assert.ok(delivery.validateObjectCatalog(catalog(item)).length);
  }
});

test("catalog absence or mismatch falls back without a fabricated preview", () => {
  assert.equal(delivery.objectEntryFor(null, entry().record), null);
  const item = entry(); item.preview.sha256 = "invalid";
  assert.equal(delivery.objectEntryFor(catalog(item), item.record), null);
  assert.equal(delivery.renderObject(null), "");
});
