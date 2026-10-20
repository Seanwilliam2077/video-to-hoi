"use strict";

// Independently authored metadata only; no challenge media or model outputs.
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const os = require("node:os");
const crypto = require("node:crypto");
const {spawnSync} = require("node:child_process");
const delivery = require("../docs/progress-board/delivery-view.js");
const SHA = "a".repeat(64);
const COMMIT = "b".repeat(40);
const URL = "https://example.test/content";
const clone = value => JSON.parse(JSON.stringify(value));

function entry() {
  const record = {
    id: "synthetic-input-declaration", episode_id: "episode_000000", stage: "inputs", iteration: "test-v1", state: "submitted", kind: "real_track1",
    created_at: "2026-10-15T12:00:00Z", contributor: "Independent fixture", reviewer: "Test reviewer", summary: "Test data only",
    delivery_url: URL, source_commit: COMMIT, coverage: {frames: 5, objects_complete: false}, upstream: [],
    artifacts: [{requirement_id: "camera", label: "Estimated camera", url: URL, sha256: SHA, bytes: 123}],
    checks: Object.fromEntries(["structural", "quality", "export"].map(key => [key, {status: "not_run", evidence_url: ""}])),
    blocker: "Full Inputs incomplete", next_action: "Run the receiving Body environment"
  };
  return {
    record, episode_id: record.episode_id, profile: "sam3d_body_masks_camera.v1", frame_count: 5,
    raw_verified: true, body_inputs_verified: true, native_readers_passed: true, body_execution_completed: false,
    camera: {fx: 900, fy: 901, cx: 320, cy: 240, width: 640, height: 480},
    review: {sample_frame_ids: [0, 4], method: "machine_assisted_visual", findings: "Sampled masks reviewed", sampling_limitation: "Sampled review does not verify every boundary."},
    archive: {url: URL, bytes: 1234, sha256: SHA, episode_path: "body-inputs/episode_000000/"}, source_video_url: URL,
    evidence: [{id: "receipt", label: "Verification receipt", url: URL, sha256: SHA, bytes: 98}],
    files: [
      {path: "intrinsics.json", group: "camera", kind: "json", url: URL, sha256: SHA, bytes: 123},
      {path: "human_masks/", group: "masks", kind: "directory", archive_path: "body-inputs/episode_000000/human_masks/"}
    ],
    previews: [{url: "previews/episode_000000-review.jpg", sha256: SHA, kind: "mask_review", label: "Actual sampled mask review"}]
  };
}
const catalog = item => ({schema_version: "hoi_delivery_catalog.v1", delivery_revision: COMMIT, entries: [item || entry()]});

test("browser and CommonJS expose the same pure rendering API", () => {
  const context = {URL: globalThis.URL};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../docs/progress-board/delivery-view.js"), "utf8"), context);
  assert.deepEqual(Object.keys(context.HOIDelivery), Object.keys(delivery));
  assert.deepEqual(delivery.validateCatalog(catalog()), []);
});

test("catalog matches the complete record independent of object key order, never just its ID", () => {
  const value = catalog();
  const reversed = Object.fromEntries(Object.entries(value.entries[0].record).reverse());
  assert.equal(delivery.entryFor(value, reversed), value.entries[0]);
  for (const mutate of [
    item => { item.summary += " changed"; },
    item => { item.iteration = "different-v2"; },
    item => { item.source_commit = "c".repeat(40); },
    item => { item.artifacts[0].sha256 = "d".repeat(64); },
    item => { item.checks.structural.status = "pass"; }
  ]) {
    const record = clone(value.entries[0].record);
    mutate(record);
    assert.equal(delivery.entryFor(value, record), null);
  }
});

test("catalog validation fails closed for malformed shapes and invalid evidence", () => {
  for (const value of [null, [], 5, false, {}, {schema_version: "hoi_delivery_catalog.v1", entries: []}]) {
    assert.ok(delivery.validateCatalog(value).length);
    assert.equal(delivery.entryFor(value, entry().record), null);
  }
  for (const mutate of [
    item => { item.frame_count = 6; },
    item => { item.raw_verified = "true"; },
    item => { item.body_inputs_verified = true; item.raw_verified = false; },
    item => { item.camera.fx = -1; },
    item => { item.review.sample_frame_ids = [5]; },
    item => { item.review.sample_frame_ids = [0, 0]; },
    item => { item.archive.episode_path = "../episode_000000"; },
    item => { item.files = [null]; },
    item => { item.evidence[0].sha256 = "bad"; },
    item => { item.record.artifacts[0].url = "javascript:alert(1)"; }
  ]) {
    const value = catalog();
    mutate(value.entries[0]);
    assert.ok(delivery.validateCatalog(value).length);
    assert.equal(delivery.entryFor(value, value.entries[0].record), null);
    assert.doesNotMatch(delivery.renderInputs(value.entries[0], "en"), /class="delivery-ready"/);
  }
  const duplicate = catalog();
  duplicate.entries.push(clone(duplicate.entries[0]));
  assert.match(delivery.validateCatalog(duplicate).join(" "), /Duplicate/);
});

test("preview sources are restricted to matching local reviewed JPEG assets", () => {
  for (const url of [
    "https://example.test/preview.jpg", "//example.test/preview.jpg", "javascript:alert(1)",
    "../previews/episode_000000-review.jpg", "previews/../secret.jpg", "previews/episode_000000-review.svg",
    "previews/episode_000001-review.jpg", "previews/episode_000000-review.jpg?query=1",
    "previews/episode_000000-review.jpg#fragment", "previews/episode_000000-review.jpg\" onerror=\"alert(1)"
  ]) {
    const value = entry();
    value.previews[0].url = url;
    assert.match(delivery.validateCatalog(catalog(value)).join(" "), /Preview/, url);
    assert.doesNotMatch(delivery.renderInputs(value), /<img/);
  }
});

test("unsafe links never become clickable and all text is HTML escaped", () => {
  for (const url of ["javascript:alert(1)", "data:text/html,unsafe", "file:///C:/secret", "https://user:pass@example.test", "https://@example.test", "https:\\example.test", "https://%65xample.test", "http://127.1", "//example.test/x", "https://example.test/\nfoo"]) {
    const value = entry();
    value.source_video_url = url;
    assert.ok(delivery.validateCatalog(catalog(value)).length, url);
    const record = value.record;
    record.artifacts[0].url = url;
    const html = delivery.renderArtifacts(record, "inputs", "en");
    assert.doesNotMatch(html, /href="(?:javascript:|data:|file:|\/\/|https:\/\/user|https:\/\/@)/);
    assert.match(html, /Incomplete registration/);
  }
  const value = entry();
  value.review.findings = '<img src=x onerror="alert(1)">';
  value.files[0].path = '<script>alert("x")</script>.json';
  value.evidence[0].label = "<svg onload='x'>";
  value.previews[0].label = '" onerror="alert(1)';
  const html = delivery.renderInputs(value, "en");
  assert.match(html, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
  assert.doesNotMatch(html, /<script|<svg|<img src=x|alt="" onerror=/);
  value.record.artifacts[0].label = "<script>bad</script>";
  value.record.next_action = '<img onerror="bad">';
  assert.doesNotMatch(delivery.renderArtifacts(value.record, "inputs"), /<script>|<img onerror/);
});

test("Body readiness does not become full Inputs acceptance or reconstructed output", () => {
  const html = delivery.renderInputs(entry(), "en");
  assert.match(html, /Body profile verification and full Inputs acceptance are separate/);
  assert.match(html, /structural: not_run/);
  assert.match(html, /Object masks/);
  assert.match(html, /Not in this package/);
  assert.match(html, /Pending Human output/);
  assert.match(html, /Body execution/);
  assert.match(html, /Not run/);
  assert.match(html, /not ground-truth calibration/);
  assert.match(html, /loading="lazy"/);
  assert.match(html, /body-inputs\/episode_000000\/human_masks\//);
  for (const id of ["raw", "body", "camera", "review", "package", "object", "depth"]) assert.match(html, new RegExp(`id="delivery-${id}"`));
});

test("selected first preview sheets do not imply that every reviewed frame is visible", () => {
  const value = entry();
  value.frame_count = value.record.coverage.frames = 500;
  value.review.sample_frame_ids = [0, 416, 432];
  const html = delivery.renderInputs(value, "en");
  assert.match(html, /0, 416, 432/);
  assert.match(html, /Selected first sheet/);
  assert.match(html, /do not show every sampled frame listed above/);
  assert.match(html, /frame IDs describe the full review scope/);
  assert.match(html, /body-inputs\/episode_000000\/mask-qa\//);
  assert.match(html, /Download package to inspect all sheets/);
  const chinese = delivery.renderInputs(value, "zh");
  assert.match(chinese, /选取的首张复查拼图/);
  assert.match(chinese, /其余 QA 拼图请在下载包中查看/);
});

test("review images precede an expandable exact original report and previews localize by kind", () => {
  const value = entry();
  value.review.findings = "Original observations <tag> with detailed known limitations.";
  value.review.sampling_limitation = "Original sampling limitation & caveat.";
  value.previews.push({url: "previews/episode_000000-targets.jpg", sha256: SHA, kind: "target_review", label: "Original target preview"});
  const html = delivery.renderInputs(value, "en");
  assert.match(html, /<details class="delivery-review-report"><summary>Review findings and limitations \(original report\)/);
  assert.ok(html.indexOf('<div class="delivery-previews">') < html.indexOf('class="delivery-review-report"'));
  assert.match(html, /Original observations &lt;tag&gt; with detailed known limitations\./);
  assert.match(html, /Original sampling limitation &amp; caveat\./);
  assert.match(html, /Sampled review does not verify every frame or boundary/);
  const chinese = delivery.renderInputs(value, "zh");
  assert.match(chinese, /复查发现与限制（原始报告）/);
  assert.match(chinese, /alt="人体 mask 全画面抽样复查"/);
  assert.match(chinese, /alt="人体目标局部抽样复查"/);
  assert.match(chinese, /Original observations &lt;tag&gt;/);
});

test("every stage offers its actual registered content or explicit missing outputs and next action", () => {
  for (const [stage, keys] of Object.entries({
    inputs: ["camera", "masks", "depth"], human: ["human", "depth_scale"], objects: ["object_record", "mesh"],
    motion: ["transforms", "confidence"], refine: ["refined_human", "refined_objects"], export: ["metadata", "parquet", "mhr_arrays", "meshes"]
  })) {
    const html = delivery.renderArtifacts(null, stage, "en");
    for (const key of keys) assert.ok(html.includes(`<code>${key}</code>`));
    assert.match(html, /Not delivered/);
    assert.match(html, /Next step/);
    assert.doesNotMatch(html, /delivery-artifact-present/);
  }
  const html = delivery.renderArtifacts(entry().record, "inputs", "en");
  assert.match(html, /Open submitted content/);
  assert.match(html, /123 bytes/);
  assert.match(html, new RegExp(SHA));
  assert.match(html, /acceptance still depends on the stage checks/);
  assert.match(html, /Registered/);
  assert.match(html, /Not delivered/);
});

test("catalog mutation is revalidated instead of receiving stale cached readiness", () => {
  const value = catalog();
  assert.ok(delivery.entryFor(value, value.entries[0].record));
  value.entries[0].previews[0].url = "https://untrusted.test/changed.jpg";
  assert.equal(delivery.entryFor(value, value.entries[0].record), null);
});

function humanEntry() {
  const base = "https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-independent-test";
  const asset = name => ({url: `${base}/independent-test-ep00-${name}`, bytes: 123, sha256: SHA});
  const archive = {...asset("review.zip"), url: `${base}/independent-test-ep00.zip`};
  const evidence = Object.fromEntries(["checks", "depth_scale", "selection", "input_bindings"].map(role => [role, asset(`${role}.json`)]));
  const record = {...entry().record, id: "independent-test-ep00", stage: "human", coverage: {frames: 5, objects_complete: null},
    artifacts: [{requirement_id: "human", label: "Actual native Human package", ...archive}, {requirement_id: "depth_scale", label: "Scale remains unverified", ...evidence.depth_scale}],
    upstream: [{stage: "inputs", record_id: "synthetic-input-declaration"}],
    checks: {structural: {status: "pass", evidence_url: evidence.checks.url}, quality: {status: "not_run", evidence_url: ""}, export: {status: "not_run", evidence_url: ""}}};
  return {record, episode_id: record.episode_id, profile: "native_mhr_mediapipe_review.v1", frame_count: 5,
    source_video_sha256: SHA, mhr_model_sha256: SHA, archive, selected_candidate: "baseline",
    selection_reason: "Independent test: retain baseline without adequate local evidence", quality_accepted: false, shared_scene_scale_verified: false, official_export_accepted: false,
    evidence, previews: [
      ...["baseline_full_body", "refined_full_body", "hands_comparison"].map(role => ({role, ...asset(`${role}.mp4`), content_type: "video/mp4", coverage: {kind: "full_sequence", original_frame_count: 5}})),
      {role: "baseline_contact_sheet", ...asset("sample.jpg"), content_type: "image/jpeg", coverage: {kind: "sampled", original_frame_ids: [0, 4]}}
    ], hand_guards: ["left", "right"].map(side => ({side, heldout_observed_frames: null, sufficient: false, before_px: null, after_px: null, nonregression: null}))};
}
const humanCatalog = item => ({schema_version: "hoi_human_catalog.v1", entries: item ? [item] : []});

test("Human previews bind the full submitted record and preserve baseline selection and limitations", () => {
  const value = humanEntry();
  assert.deepEqual(delivery.validateHumanCatalog(humanCatalog(value)), []);
  assert.equal(delivery.humanEntryFor(humanCatalog(value), clone(value.record)), value);
  const html = delivery.renderHuman(value, "en");
  assert.equal((html.match(/<video /g) || []).length, 3);
  assert.match(html, /preload="none"/);
  assert.doesNotMatch(html, /autoplay/);
  assert.match(html, /Selected local candidate.*Fixed-body baseline/);
  assert.match(html, /not accepted reconstructions/);
  assert.match(html, /previews do not change dashboard status/);
  assert.match(html, /Insufficient; keep uncertainty/);
  assert.match(html, /Sampled original frame IDs: 0, 4/);
  const zh = delivery.renderHuman(value, "zh");
  assert.match(zh, /固定身体基线/);
  assert.match(zh, /尚未通过重建验收/);
  assert.match(zh, /证据不足，保留不确定性/);
  for (const mutate of [
    r => {r.summary += " newer";}, r => {r.id += "-next";}, r => {r.iteration += "-next";},
    r => {r.artifacts[0].sha256 = "c".repeat(64);}, r => {r.upstream[0].record_id += "-next";},
    r => {r.state = "accepted";}, r => {r.checks.quality.status = "pass";}
  ]) {
    const next = clone(value.record); mutate(next);
    assert.equal(delivery.humanEntryFor(humanCatalog(value), next), null);
  }
});

test("Human preview assets fail closed for unsafe URLs, broken bindings and unsupported readiness", () => {
  for (const mutate of [
    e => {e.previews[0].url = "http://127.0.0.1/private.mp4";},
    e => {e.previews[0].url = "https://example.test/clip.mp4";},
    e => {e.previews[0].url += "?token=secret";},
    e => {e.previews[0].url = e.previews[0].url.replace("result-independent-test", "other-release");},
    e => {e.previews[0].url = null;},
    e => {e.previews[0].coverage.original_frame_count = 4;},
    e => {e.previews[3].coverage.original_frame_ids = [5];},
    e => {e.previews[3].coverage.original_frame_ids = [4, 0];},
    e => {e.previews.pop();}, e => {e.previews.push(clone(e.previews[0]));},
    e => {e.evidence.depth_scale.sha256 = "c".repeat(64);},
    e => {e.archive.sha256 = "c".repeat(64);},
    e => {e.record.artifacts.push(null);},
    e => {e.quality_accepted = true;},
    e => {e.shared_scene_scale_verified = true;},
    e => {e.hand_guards[0].before_px = Infinity;},
    e => {e.hand_guards[0].sufficient = true;},
    e => {e.injected = true;}
  ]) {
    const value = humanEntry(); mutate(value);
    assert.ok(delivery.validateHumanCatalog(humanCatalog(value)).length);
    assert.equal(delivery.renderHuman(value), "");
    assert.equal(delivery.humanEntryFor(humanCatalog(value), value.record), null);
  }
  assert.equal(delivery.renderHuman(null), "");
  assert.deepEqual(delivery.validateHumanCatalog(humanCatalog()), []);
  const duplicate = humanCatalog(humanEntry()); duplicate.entries.push(clone(duplicate.entries[0]));
  assert.match(delivery.validateHumanCatalog(duplicate).join(" "), /Duplicate/);
  const value = humanEntry(); value.selection_reason = '<img src=x onerror="bad">';
  assert.doesNotMatch(delivery.renderHuman(value, "en"), /<img src=x/);
});

test("Human catalog builder binds actual local bytes, publication verification and native evidence", () => {
  // Invented metadata/transport bytes only: no challenge video or model is read.
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "independent-human-catalog-"));
  try {
    const seed = humanEntry(), base = seed.archive.url.slice(0, seed.archive.url.lastIndexOf("/"));
    const published = {};
    const write = (name, value) => {
      const bytes = Buffer.from(typeof value === "string" ? value : JSON.stringify(value));
      const filename = path.join(directory, name); fs.writeFileSync(filename, bytes);
      const asset = {name, path: filename, bytes: bytes.length, sha256: crypto.createHash("sha256").update(bytes).digest("hex")};
      published[name] = {...asset, verified: true, url: `${base}/${name}`};
      return asset;
    };
    const ref = asset => ({url: `${base}/${asset.name}`, bytes: asset.bytes, sha256: asset.sha256});
    const checks = {schema_version: "human_batch_episode_checks.v1", track: 1, episode: 0, frames: 5,
      source_video_sha256: SHA, mhr_model_sha256: SHA, structural_checks_passed: true, reconstruction_quality_accepted: false,
      selected_candidate: "baseline", selection_reason: seed.selection_reason, native_results: {baseline: {parameters_sha256: SHA}, refined: {parameters_sha256: SHA}},
      hand_heldout_nonregression: Object.fromEntries(seed.hand_guards.map(({side, ...guard}) => [side, guard]))};
    const scale = {episode: 0, source_video_sha256: SHA, status: "unverified_monocular_scale", metric_scale_verified: false, shared_human_object_scale_verified: false};
    const selection = {episode: 0, frames: 5, selected_candidate: "baseline", selection_reason: seed.selection_reason, quality_accepted: false, selected_is_exact_candidate_copy: true, sha256: SHA};
    const bindings = {episode: 0, track: 1, video_frames: 5, video_sha256: SHA, mhr_model_sha256: SHA, source_commit: COMMIT};
    const data = {checks, "depth-scale": scale, selection, "input-bindings": bindings};
    const assets = Object.entries(data).map(([role, value]) => write(`${seed.record.id}-${role}.json`, value));
    const archive = write(`${seed.record.id}.zip`, "Independent synthetic archive identity, not real media");
    const previews = seed.previews.map(item => ({...write(item.url.split("/").at(-1), "Independent synthetic media transport identity"), role: item.role,
      content_type: item.content_type, coverage: item.coverage, episode: 0, episode_frames: 5, source_video_sha256: SHA, mhr_model_sha256: SHA}));
    const record = clone(seed.record);
    record.artifacts[0] = {requirement_id: "human", label: "Synthetic archive fixture", ...ref(archive)};
    record.artifacts[1] = {requirement_id: "depth_scale", label: "Synthetic unverified scale fixture", ...ref(assets[1])};
    record.checks.structural.evidence_url = ref(assets[0]).url;
    const result = {episode: 0, frames: 5, source_commit: COMMIT, selected_candidate: "baseline", source_video_sha256: SHA, mhr_model_sha256: SHA, assets, archive, previews};
    const input = path.join(directory, "fixture.json");
    const invoke = () => {
      fs.writeFileSync(input, JSON.stringify({package: result, record, base, published}));
      return spawnSync(process.env.PYTHON || "python", ["-c", "import importlib.util,json,sys; s=importlib.util.spec_from_file_location('catalog',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);v=json.load(open(sys.argv[2],encoding='utf-8'));print(json.dumps(m.build_entry(v['package'],v['record'],v['base'],v['published'])))", path.join(__dirname, "../tools/build_human_catalog.py"), input], {encoding: "utf8"});
    };
    const valid = invoke();
    assert.equal(valid.status, 0, valid.stderr);
    assert.deepEqual(delivery.validateHumanCatalog(humanCatalog(JSON.parse(valid.stdout))), []);
    fs.appendFileSync(previews[0].path, " modified after upload");
    const changed = invoke();
    assert.notEqual(changed.status, 0);
    assert.match(changed.stderr, /Local published asset bytes changed/);
    fs.writeFileSync(previews[0].path, "Independent synthetic media transport identity");
    published[previews[0].name].verified = false;
    const notPublished = invoke();
    assert.notEqual(notPublished.status, 0);
    assert.match(notPublished.stderr, /Publication receipt does not verify exact asset bytes/);
    published[previews[0].name].verified = true;
    result.previews[0].source_video_sha256 = "c".repeat(64);
    const wrongSource = invoke();
    assert.notEqual(wrongSource.status, 0);
    assert.match(wrongSource.stderr, /Preview provenance differs/);
  } finally {
    assert.equal(path.dirname(path.resolve(directory)), path.resolve(os.tmpdir()));
    assert.ok(path.basename(directory).startsWith("independent-human-catalog-"));
    fs.rmSync(directory, {recursive: true, force: true});
  }
});
