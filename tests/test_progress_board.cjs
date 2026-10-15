"use strict";

// Independently authored metadata and declarations only. No media or model data.
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const HOI = require("../docs/progress-board/core.js");

const SHA = "a".repeat(64);
const COMMIT = "b".repeat(40);
const REQUIREMENTS = {inputs: ["camera", "masks", "depth"], human: ["human", "depth_scale"], objects: ["object_record", "mesh"], motion: ["transforms", "confidence"], refine: ["refined_human", "refined_objects"], export: ["metadata", "parquet", "mhr_arrays", "meshes"]};
const STAGE_IDS = Object.keys(REQUIREMENTS);
const URL = "https://example.test/evidence";
const clone = value => JSON.parse(JSON.stringify(value));

function board(records = []) {
  return {
    schema_version: "hoi_progress.v1", updated_at: "2026-10-15T10:00:00Z",
    source: {inventory_path: "docs/track1-videos/index.html#DATA", repository: "https://github.com/Seanwilliam2077/video-to-hoi", revision: COMMIT, scope: "Independent synthetic unit-test declarations"},
    episodes: Array.from({length: 30}, (_, ep) => ({id: `episode_${String(ep).padStart(6, "0")}`, ep, frames: 5, seconds: 1, object: "test_cube", object_name: "Synthetic cube", camera: "synthetic", task: "Independent test", poster_url: URL, video_url: `${URL}/track_1/videos/episode_${String(ep).padStart(6, "0")}.mp4`})),
    stages: STAGE_IDS.map((id, index) => ({id, module: "platform", name: id, label: id, requirements: REQUIREMENTS[id].map(key => ({id: key, label: key})), dependencies: STAGE_IDS.slice(0, index)})),
    records, capabilities: []
  };
}

function record(id, stage = "inputs", overrides = {}) {
  return {
    id, episode_id: "episode_000000", stage, iteration: "baseline-v1", state: "planned", kind: "real_track1", created_at: "2026-10-15T09:00:00Z",
    contributor: "Test executor", reviewer: "", summary: "Independent test declaration", delivery_url: "", source_commit: "",
    coverage: {frames: null, objects_complete: null}, upstream: [], artifacts: [],
    checks: Object.fromEntries(["structural", "quality", "export"].map(key => [key, {status: "not_run", evidence_url: ""}])),
    blocker: "", next_action: "Collect evidence", ...overrides
  };
}

function checked(id, stage = "inputs", parents = [], overrides = {}) {
  const value = record(id, stage, {
    state: "checked", delivery_url: URL, source_commit: COMMIT, coverage: {frames: 5, objects_complete: true},
    upstream: parents.map(parent => ({stage: parent.stage, record_id: parent.id})),
    artifacts: REQUIREMENTS[stage].map(key => ({requirement_id: key, label: key, url: `${URL}/${key}`, sha256: SHA, bytes: 123})),
    ...overrides
  });
  value.checks.structural = {status: "pass", evidence_url: URL};
  return value;
}

test("the browser global exposes the same public API without CommonJS", () => {
  const sandbox = {URL: globalThis.URL};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../docs/progress-board/core.js"), "utf8"), sandbox);
  assert.deepEqual(Object.keys(sandbox.HOI), Object.keys(HOI));
  assert.equal(sandbox.HOI.latestFor([], "episode_000000", "inputs", "baseline-v1"), null);
});

test("empty catalog has thirty episodes and zero inferred delivery progress", () => {
  assert.deepEqual(HOI.validateBoard(board()), []);
  assert.equal(HOI.latestRecords([]).size, 0);
  assert.equal(HOI.latestFor([], "episode_000000", "inputs"), null);
  assert.equal(HOI.staleRecordIds([]).size, 0);
});

test("catalog rejects unknown source, duplicate episodes and weakened requirements", () => {
  const value = board();
  value.episodes[29] = clone(value.episodes[0]);
  value.source.inventory_path = "track_2/metadata";
  value.stages[0].requirements.pop();
  const errors = HOI.validateBoard(value).join(" ");
  assert.match(errors, /Duplicate episode/);
  assert.match(errors, /Track 1 inventory/);
  assert.match(errors, /requirements or dependencies/);
});

test("draft declarations allow unknown coverage but reject unknown fields and stages", () => {
  assert.deepEqual(HOI.validateRecord(record("draft"), board()), []);
  const value = record("draft", "native-05", {extra: true});
  assert.match(HOI.validateRecord(value, board()).join(" "), /fields.*exactly/);
  assert.match(HOI.validateRecord(value, board()).join(" "), /Unknown or invalid stage/);
});

test("dangerous URLs, credential URLs, malformed hashes and unsafe IDs are rejected", () => {
  for (const bad of ["javascript:alert(1)", "file:///C:/secret", "https://user:password@example.test", "https://@example.test", "https:///example.test", "https://example.test/\nunsafe", "https:\\example.test", "https://example.test:0/", "//example.test/file", "https://%65xample.test", "https://exa<mple.test", "https://example.123", "http://127.1", "http://999.1.1.1", "http://1.2.3.4."]) {
    assert.ok(HOI.validateRecord(record("draft", "inputs", {delivery_url: bad}), board()).length, bad);
  }
  for (const id of ["../escape", "CON", "nul.txt", "record.", "A", "x".repeat(97)]) {
    assert.match(HOI.validateRecord(record(id), board()).join(" "), /identifier/);
  }
  const value = checked("valid");
  value.artifacts[0].sha256 = "bad";
  assert.match(HOI.validateRecord(value, board()).join(" "), /sha256/);
  assert.deepEqual(HOI.validateRecord(record("upper-scheme", "inputs", {delivery_url: "HTTPS://example.test/evidence"}), board()), []);
  for (const delivery_url of ["http://localhost:8000", "http://1.2.3.4", "https://[::1]/", "http://0xgarbage", "https://xn--bcher-kva.example"]) {
    assert.deepEqual(HOI.validateRecord(record("host", "inputs", {delivery_url}), board()), [], delivery_url);
  }
});

test("malformed primitive shapes return errors rather than throwing", () => {
  for (const value of [null, [], false, 42, {}, {coverage: null}, {id: "x", upstream: [null], artifacts: [null], checks: null}]) {
    assert.doesNotThrow(() => HOI.validateRecord(value, board()));
    assert.ok(HOI.validateRecord(value, board()).length);
  }
  assert.ok(HOI.validateBoard({records: null, episodes: [null], stages: [null]}).length);
});

test("invalid calendar dates and timezone-less timestamps are rejected", () => {
  for (const created_at of ["2026-02-30T10:00:00Z", "2026-10-15T24:00:00Z", "2026-10-15T10:00:00", "2026-10-15T10:00:00+00:60", "2026-10-15T10:00:00.123456Z", "20261015T100000Z", "not a date"]) {
    assert.match(HOI.validateRecord(record("date", "inputs", {created_at}), board()).join(" "), /timestamp/);
  }
});

test("timestamps compare timezone-adjusted milliseconds without losing wire precision", () => {
  const older = record("z", "inputs", {created_at: "2026-10-15T10:00:00.001Z"});
  const newer = record("a", "inputs", {created_at: "2026-10-15T18:00:00.002+08:00"});
  assert.deepEqual(HOI.validateRecord(newer, board()), []);
  assert.equal(HOI.latestFor([newer, older], older.episode_id, "inputs").id, "a");
});

test("artifact byte counts and coverage cannot exceed exact JSON number precision", () => {
  const value = checked("large-count");
  value.artifacts[0].bytes = Number.MAX_SAFE_INTEGER + 1;
  value.coverage.frames = Number.MAX_SAFE_INTEGER + 1;
  const errors = HOI.validateRecord(value, board()).join(" ");
  assert.match(errors, /Artifact bytes/);
  assert.match(errors, /Coverage requires/);
});

test("synthetic or engineering work never qualifies as checked video progress", () => {
  for (const kind of ["synthetic", "engineering"]) {
    const value = checked("fixture", "inputs", [], {kind});
    assert.match(HOI.validateRecord(value, board()).join(" "), /Only real_track1/);
  }
});

test("checked requires full artifact inventory, full coverage and structural evidence", () => {
  const value = checked("inputs-complete");
  assert.deepEqual(HOI.validateRecord(value, board()), []);
  assert.deepEqual(HOI.missingRequirements(value, board().stages[0]), []);
  value.artifacts.pop();
  value.coverage.frames = 4;
  value.checks.structural.evidence_url = "";
  const errors = HOI.validateRecord(value, board()).join(" ");
  assert.match(errors, /Incomplete required artifacts: depth/);
  assert.match(errors, /every original frame/);
  assert.match(errors, /structural.*evidence/);
  assert.deepEqual(HOI.missingRequirements(value, board().stages[0]), ["depth"]);
});

test("submitted state does not imply passing checks, and failing checks need evidence", () => {
  const value = record("submitted", "inputs", {state: "submitted"});
  assert.deepEqual(HOI.validateRecord(value, board()), []);
  value.checks.quality.status = "fail";
  assert.match(HOI.validateRecord(value, board()).join(" "), /quality.*evidence/);
});

test("accepted requires separate reviewer, quality and final export evidence", () => {
  const rows = [];
  for (const stage of STAGE_IDS) rows.push(checked(`${stage}-1`, stage, rows));
  const final = rows.at(-1);
  final.state = "accepted";
  let errors = HOI.validateRecord(final, board(rows), rows).join(" ");
  assert.match(errors, /reviewer/);
  assert.match(errors, /quality pass/);
  assert.match(errors, /export pass/);
  final.reviewer = "Test reviewer";
  final.checks.quality = {status: "pass", evidence_url: URL};
  final.checks.export = {status: "pass", evidence_url: URL};
  assert.deepEqual(HOI.validateBoard(board(rows)), []);
});

test("upstreams bind exact stage and episode, every dependency and execution order", () => {
  const inputs = checked("inputs-1");
  const human = checked("human-1", "human", [inputs]);
  assert.deepEqual(HOI.validateRecord(human, board([inputs, human])), []);
  inputs.episode_id = "episode_000001";
  inputs.created_at = "2026-10-15T10:00:00Z";
  let errors = HOI.validateRecord(human, board([inputs, human])).join(" ");
  assert.match(errors, /different stage or episode/);
  assert.match(errors, /newer than its consumer/);
  human.upstream = [];
  assert.match(HOI.validateRecord(human, board([inputs, human])).join(" "), /Required upstream stage/);
});

test("duplicate IDs, missing/self references and cycles cannot enter a board", () => {
  const a = record("a");
  const b = record("b", "human", {upstream: [{stage: "inputs", record_id: "a"}]});
  a.upstream = [{stage: "human", record_id: "b"}];
  assert.match(HOI.validateBoard(board([a, b])).join(" "), /cycle/i);
  a.upstream = [{stage: "inputs", record_id: "a"}];
  assert.match(HOI.validateBoard(board([a])).join(" "), /reference itself/);
  b.upstream[0].record_id = "missing";
  assert.match(HOI.validateBoard(board([b])).join(" "), /Missing upstream/);
  assert.match(HOI.validateBoard(board([record("a"), record("a")])).join(" "), /Duplicate record id/);
});

test("merge is immutable, idempotent and atomic when a later item conflicts", () => {
  const original = record("original");
  const existing = [original];
  const reordered = Object.fromEntries(Object.entries(original).reverse());
  assert.deepEqual(HOI.mergeRecords(existing, [reordered]).records, existing);
  assert.deepEqual(HOI.mergeRecords(existing, [reordered]).errors, []);
  const conflict = {...clone(original), summary: "replacement"};
  const merged = HOI.mergeRecords(existing, [record("new"), conflict]);
  assert.match(merged.errors.join(" "), /Conflicting immutable record/);
  assert.deepEqual(merged.records, existing);
  assert.equal(existing.length, 1);
  assert.equal(original.summary, "Independent test declaration");
  assert.equal(HOI.mergeRecords([], [original, clone(original)]).records.length, 1);
});

test("import accepts only a raw record or exact versioned record envelope", () => {
  const value = record("draft");
  assert.deepEqual(HOI.parseImport(JSON.stringify(value)), {records: [value], errors: []});
  assert.deepEqual(HOI.parseImport("\uFEFF" + JSON.stringify({schema_version: "hoi_progress.v1", records: [value]})), {records: [value], errors: []});
  for (const payload of [[], [value], board(), {schema_version: "unknown", records: []}, {schema_version: "hoi_progress.v1", records: [], stages: []}, {schema_version: "hoi_progress.v1", records: {}}]) {
    const result = HOI.parseImport(JSON.stringify(payload));
    assert.ok(result.errors.length, JSON.stringify(payload));
    assert.deepEqual(result.records, []);
  }
});

test("import rejects duplicate JSON keys including escaped duplicates at any depth", () => {
  for (const json of ['{"schema_version":"hoi_progress.v1","records":[],"records":[]}', '{"schema_version":"hoi_progress.v1","records":[{"id":"a","\\u0069d":"b"}]}', '{"schema_version":"hoi_progress.v1","records":[{"checks":{"quality":{},"quality":{}}}]}']) {
    const result = HOI.parseImport(json);
    assert.match(result.errors.join(" "), /Duplicate JSON key/);
    assert.deepEqual(result.records, []);
  }
  const value = record("strings", "inputs", {summary: 'A quoted "records": [] string, backslash \\, and braces {} are content.'});
  assert.deepEqual(HOI.parseImport(JSON.stringify({schema_version: "hoi_progress.v1", records: [value]})).errors, []);
  assert.ok(HOI.parseImport("{broken}").errors.length);
});

test("checked consumers cannot mix individually valid but incoherent upstream chains", () => {
  const inputA = checked("inputs-a");
  const inputB = checked("inputs-b", "inputs", [], {iteration: "candidate-v2"});
  const humanB = checked("human-b", "human", [inputB], {iteration: "candidate-v2"});
  const mixed = checked("objects-mixed", "objects", [inputA, humanB], {iteration: "candidate-v2"});
  const rows = [inputA, inputB, humanB, mixed];
  assert.equal(HOI.staleRecordIds(rows).size, 0);
  assert.match(HOI.validateRecord(mixed, board(rows)).join(" "), /Incoherent upstream chain/);
  humanB.upstream = [{stage: "inputs", record_id: inputA.id}];
  assert.deepEqual(HOI.validateBoard(board(rows)), []);
});

test("latest selection is deterministic per episode, stage and iteration", () => {
  const a = record("a");
  const z = record("z");
  const newer = record("new", "inputs", {iteration: "candidate-v2", created_at: "2026-10-15T10:00:00Z"});
  const rows = [newer, z, a];
  assert.equal(HOI.latestFor(rows, a.episode_id, "inputs", "baseline-v1").id, "z");
  assert.equal(HOI.latestRecords(rows, "candidate-v2").get("episode_000000:inputs").id, "new");
  assert.equal(HOI.latestRecords(rows).get("episode_000000:inputs").id, "new");
});

test("staleness propagates without rewriting history and permits cross-iteration baseline reuse", () => {
  const inputs = checked("inputs-1");
  const human = checked("human-1", "human", [inputs], {iteration: "candidate-v2"});
  const objects = checked("objects-1", "objects", [inputs, human], {iteration: "candidate-v2"});
  const unrelated = record("inputs-other", "inputs", {iteration: "other", created_at: "2026-10-15T11:00:00Z"});
  const rows = [inputs, human, objects, unrelated];
  assert.equal(HOI.staleRecordIds(rows).size, 0);
  const failedNewAttempt = record("inputs-2", "inputs", {state: "failed", created_at: "2026-10-15T12:00:00Z"});
  assert.deepEqual([...HOI.staleRecordIds([...rows, failedNewAttempt])].sort(), ["human-1", "objects-1"]);
  assert.equal(human.state, "checked");
  assert.equal(objects.upstream[0].record_id, "inputs-1");
});

function acceptedChain() {
  const rows = [];
  for (const stage of STAGE_IDS) {
    const value = checked(`${stage}-accepted`, stage, rows, {state: "accepted", reviewer: "Test reviewer", iteration: stage === "export" ? "candidate-v2" : "baseline-v1"});
    value.checks.quality = {status: "pass", evidence_url: URL};
    if (stage === "export") value.checks.export = {status: "pass", evidence_url: URL};
    rows.push(value);
  }
  return rows;
}

test("complete-video acceptance follows an exact export closure across iterations", () => {
  const rows = acceptedChain();
  assert.deepEqual(HOI.validateBoard(board(rows)), []);
  assert.deepEqual([...HOI.acceptedEpisodeIds(rows)], ["episode_000000"]);
  assert.deepEqual([...HOI.acceptedEpisodeIds(rows, "candidate-v2")], ["episode_000000"]);
  assert.equal(HOI.acceptedEpisodeIds(rows, "baseline-v1").size, 0);
  const unrelated = {...clone(rows[0]), id: "unrelated-input", iteration: "separate-experiment"};
  assert.equal(HOI.acceptedEpisodeIds([...rows, unrelated]).size, 1);
});

test("unrelated accepted stages never combine into a complete video", () => {
  const rows = acceptedChain();
  const otherInput = {...clone(rows[0]), id: "other-input", iteration: "other"};
  rows[1].upstream = [{stage: "inputs", record_id: otherInput.id}];
  assert.equal(HOI.acceptedEpisodeIds([...rows, otherInput]).size, 0);
  rows[1].upstream = [{stage: "inputs", record_id: rows[0].id}];
  rows[2].state = "checked";
  assert.equal(HOI.acceptedEpisodeIds(rows).size, 0);
});

test("new failed export masks older acceptance only within its own iteration", () => {
  const rows = acceptedChain();
  const failed = record("export-failed", "export", {state: "failed", iteration: "candidate-v2", created_at: "2026-10-15T10:00:00Z"});
  assert.equal(HOI.acceptedEpisodeIds([...rows, failed]).size, 0);
  assert.equal(HOI.acceptedEpisodeIds([...rows, {...failed, iteration: "other"}]).size, 1);
  const newInputs = record("inputs-new", "inputs", {created_at: "2026-10-15T11:00:00Z"});
  assert.equal(HOI.acceptedEpisodeIds([...rows, newInputs]).size, 0);
});
