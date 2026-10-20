"use strict";

// Exercise the real browser app's events and rendered state. The tiny DOM models
// only the browser APIs this app uses; all metadata is independently authored.
// Layout, focus and image loading are covered by browser QA, not this harness.
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const HOI = require("../docs/progress-board/core.js");
const HOIDelivery = require("../docs/progress-board/delivery-view.js");
const APP = fs.readFileSync(path.join(__dirname, "../docs/progress-board/app.js"), "utf8");
const SHA = "a".repeat(64), COMMIT = "b".repeat(40);
const CONTENT = "https://example.test/content";
const EP = "episode_000000";
const REQUIREMENTS = {
  inputs: ["camera", "masks", "depth"], human: ["human", "depth_scale"],
  objects: ["object_record", "mesh"], motion: ["transforms", "confidence"],
  refine: ["refined_human", "refined_objects"], export: ["metadata", "parquet", "mhr_arrays", "meshes"]
};
const clone = value => JSON.parse(JSON.stringify(value));

function record(id = "baseline-inputs", overrides = {}) {
  return {
    id, episode_id: EP, stage: "inputs", iteration: "baseline-v1", state: "submitted", kind: "real_track1",
    created_at: "2026-10-15T09:00:00Z", contributor: "Independent test executor", reviewer: "",
    summary: `Evidence for ${id}`, delivery_url: CONTENT, source_commit: COMMIT,
    coverage: {frames: 5, objects_complete: false}, upstream: [],
    artifacts: [{requirement_id: "camera", label: "Camera", url: CONTENT, sha256: SHA, bytes: 123}],
    checks: Object.fromEntries(["structural", "quality", "export"].map(key => [key, {status: "not_run", evidence_url: ""}])),
    blocker: "", next_action: "Collect the next independent output", ...overrides
  };
}

function board(records = [record()]) {
  const stageIds = Object.keys(REQUIREMENTS);
  return {
    schema_version: "hoi_progress.v1", updated_at: "2026-10-15T10:00:00Z",
    source: {inventory_path: "docs/track1-videos/index.html#DATA", repository: "https://github.com/Seanwilliam2077/video-to-hoi", revision: COMMIT, scope: "Independent test declarations"},
    episodes: Array.from({length: 30}, (_, ep) => {
      const id = `episode_${String(ep).padStart(6, "0")}`;
      return {id, ep, frames: 5, seconds: 1, object: "test_cube", object_name: "Test cube", camera: "test", task: "Independent test", poster_url: CONTENT, video_url: `${CONTENT}/track_1/videos/${id}.mp4`};
    }),
    stages: stageIds.map((id, index) => ({id, module: "platform", name: id, label: id, requirements: REQUIREMENTS[id].map(key => ({id: key, label: key})), dependencies: stageIds.slice(0, index)})),
    records, capabilities: []
  };
}

function catalog(input = record()) {
  return {
    schema_version: "hoi_delivery_catalog.v1", delivery_revision: COMMIT,
    entries: [{
      record: clone(input), episode_id: EP, profile: "sam3d_body_masks_camera.v1", frame_count: 5,
      raw_verified: true, body_inputs_verified: true, native_readers_passed: true, body_execution_completed: false,
      camera: {fx: 900, fy: 901, cx: 320, cy: 240, width: 640, height: 480},
      review: {sample_frame_ids: [0, 4], method: "machine_assisted_visual", findings: "Independent test finding", sampling_limitation: "Only sampled frames."},
      archive: {url: CONTENT, bytes: 1234, sha256: SHA, episode_path: `body-inputs/${EP}`},
      source_video_url: CONTENT,
      evidence: [{id: "receipt", label: "Independent receipt", url: CONTENT, sha256: SHA, bytes: 98}],
      files: [{path: "intrinsics.json", group: "camera", kind: "json", url: CONTENT, sha256: SHA, bytes: 123}],
      previews: []
    }]
  };
}

class Element {
  constructor(id) {
    this.id = id; this.value = ""; this.hidden = false; this.open = false;
    this.dataset = {}; this.style = {}; this.listeners = new Map(); this._html = "";
    this.textContent = ""; this.classList = {toggle() {}, add() {}, remove() {}};
    this.elements = {namedItem: () => ({focus() {}})};
  }
  set innerHTML(value) {
    this._html = value;
    if (this.id.endsWith("-filter")) this.value = (value.match(/<option value="([^"]*)"/) || [,""])[1];
  }
  get innerHTML() { return this._html; }
  addEventListener(name, handler) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(handler);
  }
  async emit(name, event = {}) {
    for (const handler of this.listeners.get(name) || []) await handler(event);
  }
  showModal() { this.open = true; }
  close() { this.open = false; void this.emit("close"); }
  setAttribute() {}
  scrollIntoView() {}
  focus() {}
}

async function app({registry = board(), contents = catalog(), hash = ""} = {}) {
  assert.deepEqual(HOI.validateBoard(registry), []);
  const elements = new Map(), stored = new Map(), events = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  };
  const location = new URL(`https://example.test/progress/${hash}`);
  const network = {registry: clone(registry), contents: clone(contents), catalogFailure: false};
  const history = {
    pushState(_state, _title, url) { location.href = new URL(url, location.href).href; },
    replaceState(_state, _title, url) { location.href = new URL(url, location.href).href; }
  };
  const sandbox = {
    HOI, HOIDelivery, URL, URLSearchParams, AbortController, location, history,
    matchMedia: () => ({matches: false}),
    setTimeout: () => 1, clearTimeout() {},
    crypto: {getRandomValues(array) { array.fill(7); return array; }},
    localStorage: {getItem: key => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value)},
    FormData: class { constructor(form) { this.values = form.values; } get(name) { return this.values[name] ?? null; } },
    document: {
      documentElement: {lang: ""}, getElementById: get,
      querySelectorAll: () => [], querySelector: selector => get(selector),
      createElement: tag => new Element(tag), body: {append() {}}
    },
    window: {addEventListener(name, handler) {
      if (!events.has(name)) events.set(name, []);
      events.get(name).push(handler);
    }},
    fetch: async url => {
      const isCatalog = String(url).includes("delivery-catalog");
      if (isCatalog && network.catalogFailure) throw new Error("Independent simulated catalog outage");
      return {ok: true, json: async () => clone(isCatalog ? network.contents : network.registry)};
    }
  };
  vm.runInNewContext(APP, sandbox, {filename: "app.js"});
  // Complete the app's actual asynchronous startup fetches and rendering.
  for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
  assert.equal(get("load-error").textContent, "");
  assert.match(get("#matrix tbody").innerHTML, /data-episode="episode_000000"/);
  return {
    get, network, location, stored,
    async click(id, dataset = {}, targetId = "") {
      const target = new Element(targetId); target.dataset = dataset;
      target.closest = selector => selector === "button" || (selector.includes("data-episode") && dataset.episode) || (selector.includes("data-preparation") && dataset.preparation) ? target : null;
      await get(id).emit("click", {target});
    },
    async change(id, value) { get(id).value = value; await get(id).emit("change"); },
    async navigate(hashValue) {
      location.hash = hashValue;
      for (const handler of events.get("hashchange") || []) await handler();
    }
  };
}

const recordInUrl = value => new URLSearchParams(value.location.hash.slice(1)).get("record");
const preparationCount = (value, item) => {
  const html = value.get("preparation-cards").innerHTML;
  return Number(html.match(new RegExp(`data-preparation="${item}"[^]*?<strong>(\\d+)`))?.[1]);
};

test("an open exact record remains pinned when repository refresh introduces a newer iteration", async () => {
  const value = await app();
  await value.click("board-panel", {episode: EP, stage: "inputs"});
  assert.equal(recordInUrl(value), "baseline-inputs");
  const newer = record("new-inputs", {iteration: "next-v2", created_at: "2026-10-16T09:00:00Z"});
  value.network.registry.records.push(newer);
  await value.click("refresh");
  assert.equal(recordInUrl(value), "baseline-inputs");
  assert.match(value.get("detail-content").innerHTML, /Evidence for baseline-inputs/);
  assert.doesNotMatch(value.get("detail-content").innerHTML.split('class="detail-summary"')[1].split("</p>")[0], /Evidence for new-inputs/);
  assert.equal(preparationCount(value, "body"), 0, "the current iteration does not borrow the historical catalog");
});

test("saving then removing a draft updates both the rendered selection and shareable route", async () => {
  const value = await app();
  await value.click("board-panel", {episode: EP, stage: "inputs"});
  await value.click("detail-content", {}, "new-record");
  const form = value.get("record-form");
  form.values = {
    id: "new-draft-inputs", iteration: "next-v2", state: "planned", kind: "real_track1",
    contributor: "Independent test executor", summary: "New draft output", next_action: "Collect evidence",
    check_structural_status: "not_run", check_quality_status: "not_run", check_export_status: "not_run"
  };
  await form.emit("submit", {preventDefault() {}, currentTarget: form});
  assert.equal(value.get("form-errors").textContent, "");
  assert.equal(recordInUrl(value), "new-draft-inputs");
  assert.match(value.get("detail-content").innerHTML, /New draft output/);
  assert.equal(JSON.parse(value.stored.get("hoi-progress-drafts-v1")).records[0].id, "new-draft-inputs");
  await value.click("detail-content", {record: "new-draft-inputs"}, "remove-draft");
  assert.equal(recordInUrl(value), "baseline-inputs");
  assert.match(value.get("detail-content").innerHTML, /Evidence for baseline-inputs/);
  assert.equal(JSON.parse(value.stored.get("hoi-progress-drafts-v1")).records.length, 0);
});

test("an invalid exact record route does not silently display another record", async () => {
  const value = await app({hash: "#episode=episode_000000&stage=inputs&record=missing-record"});
  assert.equal(value.get("detail").open, false);
  assert.match(value.get("toast").textContent, /不存在|does not exist/i);
  assert.doesNotMatch(value.get("detail-content").innerHTML, /Evidence for baseline-inputs/);
  await value.click("board-panel", {episode: EP, stage: "inputs"});
  assert.equal(value.get("detail").open, true);
  assert.equal(recordInUrl(value), "baseline-inputs");
  await value.navigate("#episode=episode_000000&stage=inputs&record=missing-record");
  assert.equal(value.get("detail").open, false, "an invalid active route must hide the previous record");
  assert.match(value.get("toast").textContent, /不存在|does not exist/i);
});

test("preparation totals follow the iteration and restore historical verification only on exact match", async () => {
  const next = record("next-inputs", {iteration: "next-v2", created_at: "2026-10-16T09:00:00Z"});
  const value = await app({registry: board([record(), next])});
  assert.equal(preparationCount(value, "body"), 0);
  await value.change("iteration-filter", "baseline-v1");
  assert.equal(preparationCount(value, "body"), 1);
  assert.equal(preparationCount(value, "raw"), 1);
  await value.change("iteration-filter", "next-v2");
  assert.equal(preparationCount(value, "body"), 0);
  await value.click("board-panel", {episode: EP, stage: "inputs"});
  assert.equal(recordInUrl(value), "next-inputs");
  assert.doesNotMatch(value.get("detail-content").innerHTML, /class="delivery-ready"/);
});

test("catalog outage during refresh retains stage browsing but removes evidence-based readiness", async () => {
  const value = await app();
  assert.equal(preparationCount(value, "body"), 1);
  value.network.catalogFailure = true;
  await value.click("refresh");
  assert.equal(value.get("stat-registered").textContent, 1);
  assert.equal(preparationCount(value, "body"), 0);
  assert.equal(value.get("catalog-warning").hidden, false);
  await value.click("board-panel", {episode: EP, stage: "inputs"});
  assert.equal(value.get("detail").open, true);
  assert.match(value.get("detail-content").innerHTML, /Open submitted content|打开递交内容/);
  assert.doesNotMatch(value.get("detail-content").innerHTML, /class="delivery-ready"/);
});
