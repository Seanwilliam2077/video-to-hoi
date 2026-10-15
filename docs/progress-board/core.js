(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.HOI = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const STATES = Object.freeze(["planned", "running", "blocked", "submitted", "checked", "accepted", "failed"]);
  const KINDS = Object.freeze(["real_track1", "synthetic", "engineering"]);
  const CHECKS = ["structural", "quality", "export"];
  const CHECK_STATES = ["not_run", "pass", "fail"];
  const READY = ["checked", "accepted"];
  const ID = /^[a-z0-9][a-z0-9._-]{0,95}$/;
  const SHA256 = /^[a-f0-9]{64}$/;
  const COMMIT = /^[a-f0-9]{40}$/;
  const RESERVED = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/;
  const RECORD_FIELDS = ["id", "episode_id", "stage", "iteration", "state", "kind", "created_at", "contributor", "reviewer", "summary", "delivery_url", "source_commit", "coverage", "upstream", "artifacts", "checks", "blocker", "next_action"];
  const REQUIREMENTS = {inputs: ["camera", "masks", "depth"], human: ["human", "depth_scale"], objects: ["object_record", "mesh"], motion: ["transforms", "confidence"], refine: ["refined_human", "refined_objects"], export: ["metadata", "parquet", "mhr_arrays", "meshes"]};
  // Match Python's shared wire format; sub-millisecond dates lose ordering in JS.
  const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/;
  const object = value => value !== null && typeof value === "object" && !Array.isArray(value);
  const text = value => typeof value === "string";
  const nonempty = value => text(value) && value.trim().length > 0;
  const list = value => Array.isArray(value) ? value : [];
  const positiveInteger = value => Number.isSafeInteger(value) && value > 0;
  const episodeId = episode => episode.id;
  const requirementId = requirement => object(requirement) ? requirement.id : requirement;
  const dependencies = stage => list(stage && stage.dependencies);
  const identifier = value => text(value) && ID.test(value) && !value.endsWith(".") && !RESERVED.test(value);
  const exactFields = (value, fields) => object(value) && Object.keys(value).length === fields.length && fields.every(key => Object.hasOwn(value, key));

  function validTimestamp(value) {
    if (!text(value) || !ISO.test(value) || !Number.isFinite(Date.parse(value))) return false;
    const [year, month, day] = value.slice(0, 10).split("-").map(Number);
    const days = [31, year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    return year >= 1 && month >= 1 && month <= 12 && day >= 1 && day <= days[month - 1] && Number(value.slice(11, 13)) < 24 && Number(value.slice(14, 16)) < 60 && Number(value.slice(17, 19)) < 60;
  }

  function validUrl(value) {
    if (!text(value)) return false;
    if (value === "") return true;
    if (/[\s\x00-\x1f\\]/.test(value)) return false;
    const authority = value.match(/^https?:\/\/([^/?#]+)/i);
    if (!authority || authority[1].includes("@")) return false;
    // Use the same conservative host grammar as the Python importer. URL()
    // otherwise repairs encoded hosts and noncanonical numeric IPv4 spellings.
    const rawHost = authority[1].startsWith("[") ? authority[1].slice(1, authority[1].indexOf("]")) : authority[1].split(":")[0];
    if (!rawHost || !/^[a-z0-9._:-]+$/i.test(rawHost)) return false;
    if (!rawHost.includes(":")) {
      const last = rawHost.replace(/\.$/, "").split(".").at(-1);
      if (/^(?:[0-9]+|0x[0-9a-f]+)$/i.test(last)) {
        const parts = rawHost.split(".");
        if (parts.length !== 4 || parts.some(part => !/^(?:0|[1-9][0-9]{0,2})$/.test(part) || Number(part) > 255)) return false;
      }
    }
    try {
      const url = new URL(value);
      return ["http:", "https:"].includes(url.protocol) && Boolean(url.hostname) && !url.username && !url.password && url.port !== "0";
    } catch (_) { return false; }
  }

  function stable(value) {
    if (Array.isArray(value)) return "[" + value.map(stable).join(",") + "]";
    if (object(value)) return "{" + Object.keys(value).sort().map(key => JSON.stringify(key) + ":" + stable(value[key])).join(",") + "}";
    return JSON.stringify(value);
  }

  function compareRecords(left, right) {
    const delta = Date.parse(left.created_at) - Date.parse(right.created_at);
    if (Number.isFinite(delta) && delta !== 0) return delta;
    return String(left.id) < String(right.id) ? -1 : String(left.id) === String(right.id) ? 0 : 1;
  }

  function latestFor(records, selectedEpisode, stageId, iteration) {
    return list(records).filter(record => object(record) && record.episode_id === selectedEpisode && record.stage === stageId && (iteration === undefined || record.iteration === iteration))
      .reduce((latest, record) => latest === null || compareRecords(record, latest) > 0 ? record : latest, null);
  }

  function latestRecords(records, iteration) {
    const result = new Map();
    for (const record of list(records)) {
      if (!object(record) || (iteration !== undefined && record.iteration !== iteration)) continue;
      const key = `${record.episode_id}:${record.stage}`;
      if (!result.has(key) || compareRecords(record, result.get(key)) > 0) result.set(key, record);
    }
    return result;
  }

  function artifactComplete(artifact) {
    return object(artifact) && nonempty(artifact.url) && validUrl(artifact.url) && text(artifact.sha256) && SHA256.test(artifact.sha256) && positiveInteger(artifact.bytes);
  }

  function missingRequirements(record, stage) {
    return list(stage && stage.requirements).map(requirementId)
      .filter(id => !list(record && record.artifacts).some(artifact => object(artifact) && artifact.requirement_id === id && artifactComplete(artifact)));
  }

  function staleRecordIds(records) {
    const rows = list(records).filter(object);
    const byId = new Map(rows.map(record => [record.id, record]));
    const latest = new Map();
    const key = record => `${record.episode_id}:${record.stage}:${record.iteration}`;
    for (const record of rows) {
      if (!latest.has(key(record)) || compareRecords(record, latest.get(key(record))) > 0) latest.set(key(record), record);
    }
    const stale = new Set();
    for (const record of rows) {
      for (const ref of list(record.upstream)) {
        const parent = object(ref) ? byId.get(ref.record_id) : null;
        const newest = parent && latest.get(key(parent));
        if (!parent || !newest || newest.id !== parent.id) stale.add(record.id);
      }
    }
    let changed = true;
    while (changed) {
      changed = false;
      for (const record of rows) {
        if (!stale.has(record.id) && list(record.upstream).some(ref => object(ref) && stale.has(ref.record_id))) {
          stale.add(record.id);
          changed = true;
        }
      }
    }
    return stale;
  }

  function acceptedEpisodeIds(records, iteration) {
    const rows = list(records).filter(object);
    const byId = new Map(rows.map(record => [record.id, record]));
    const stale = staleRecordIds(rows);
    const exports = new Map();
    for (const record of rows) {
      if (record.stage !== "export" || (iteration !== undefined && record.iteration !== iteration)) continue;
      const key = `${record.episode_id}:${record.iteration}`;
      if (!exports.has(key) || compareRecords(record, exports.get(key)) > 0) exports.set(key, record);
    }
    const accepted = new Set();
    for (const final of exports.values()) {
      const bindings = new Map();
      const visiting = new Set();
      function inspect(record) {
        if (!record || record.episode_id !== final.episode_id || record.kind !== "real_track1" || record.state !== "accepted" || stale.has(record.id) || !Object.hasOwn(REQUIREMENTS, record.stage)) return false;
        if (visiting.has(record.id)) return false;
        if (bindings.has(record.stage)) return bindings.get(record.stage) === record.id;
        const expected = Object.keys(REQUIREMENTS).slice(0, Object.keys(REQUIREMENTS).indexOf(record.stage));
        const refs = list(record.upstream);
        if (refs.length !== expected.length || new Set(refs.map(ref => ref && ref.stage)).size !== expected.length || refs.some(ref => !object(ref) || !expected.includes(ref.stage))) return false;
        bindings.set(record.stage, record.id);
        visiting.add(record.id);
        for (const ref of refs) {
          const parent = byId.get(ref.record_id);
          if (!parent || parent.stage !== ref.stage || !inspect(parent)) return false;
        }
        visiting.delete(record.id);
        return true;
      }
      if (inspect(final) && bindings.size === 6) accepted.add(final.episode_id);
    }
    return accepted;
  }

  function cycleIds(records) {
    const byId = new Map(list(records).filter(object).map(record => [record.id, record]));
    const completed = new Set();
    const active = [];
    const cycles = new Set();
    function visit(id) {
      if (active.includes(id)) {
        active.slice(active.indexOf(id)).forEach(value => cycles.add(value));
        return;
      }
      if (completed.has(id) || !byId.has(id)) return;
      active.push(id);
      list(byId.get(id).upstream).forEach(ref => { if (object(ref)) visit(ref.record_id); });
      active.pop();
      completed.add(id);
    }
    byId.forEach((_, id) => visit(id));
    return cycles;
  }

  function validateRecord(record, board, allRecords) {
    const errors = [];
    if (!object(record)) return ["Record must be an object."];
    if (!exactFields(record, RECORD_FIELDS)) errors.push("Record fields do not match the schema exactly.");
    const rows = list(allRecords === undefined ? board && board.records : allRecords);
    const stages = list(board && board.stages);
    const episodes = list(board && board.episodes);
    const stage = stages.find(item => object(item) && item.id === record.stage);
    const episode = episodes.find(item => object(item) && episodeId(item) === record.episode_id);
    if (!identifier(record.id)) errors.push("Record id must be a safe lowercase identifier.");
    if (!text(record.episode_id) || !episode) errors.push("Unknown or invalid episode_id.");
    if (!text(record.stage) || !stage) errors.push("Unknown or invalid stage.");
    if (!identifier(record.iteration)) errors.push("Iteration must be a safe lowercase identifier.");
    if (!STATES.includes(record.state)) errors.push("Unknown workflow state.");
    if (!KINDS.includes(record.kind)) errors.push("Unknown record kind.");
    if (!validTimestamp(record.created_at)) errors.push("created_at must be a valid ISO timestamp with timezone.");
    for (const field of ["contributor", "reviewer", "summary", "blocker", "next_action"]) {
      if (!text(record[field])) errors.push(`${field} must be a string.`);
    }
    if (!nonempty(record.summary)) errors.push("Summary is required.");
    if (!nonempty(record.contributor)) errors.push("Actual contributor is required.");
    if (record.state === "blocked" && !nonempty(record.blocker)) errors.push("Blocked records require a blocker.");
    if (!validUrl(record.delivery_url)) errors.push("delivery_url must be an HTTP(S) URL or empty.");
    if (!text(record.source_commit) || (record.source_commit !== "" && !COMMIT.test(record.source_commit))) errors.push("source_commit must be a full 40-character commit SHA or empty.");
    if (!exactFields(record.coverage, ["frames", "objects_complete"]) || (record.coverage.frames !== null && (!Number.isSafeInteger(record.coverage.frames) || record.coverage.frames < 0)) || (record.coverage.objects_complete !== null && typeof record.coverage.objects_complete !== "boolean")) errors.push("Coverage requires null or nonnegative integer frames and null or boolean objects_complete.");
    if (!Array.isArray(record.upstream)) errors.push("upstream must be an array.");
    if (!Array.isArray(record.artifacts)) errors.push("artifacts must be an array.");
    if (!exactFields(record.checks, CHECKS)) errors.push("checks must have exactly structural, quality and export checks.");
    for (const key of CHECKS) {
      const check = object(record.checks) && record.checks[key];
      if (!exactFields(check, ["status", "evidence_url"]) || !CHECK_STATES.includes(check.status) || !validUrl(check.evidence_url)) errors.push(`${key} check requires a valid status and HTTP(S) evidence URL or empty.`);
      else if (check.status !== "not_run" && !nonempty(check.evidence_url)) errors.push(`${key} check outcome requires an evidence URL.`);
    }

    const required = new Set(list(stage && stage.requirements).map(requirementId));
    const seenArtifacts = new Set();
    for (const artifact of list(record.artifacts)) {
      if (!object(artifact)) { errors.push("Each artifact must be an object."); continue; }
      if (!exactFields(artifact, ["requirement_id", "label", "url", "sha256", "bytes"])) errors.push("Artifact fields do not match the schema exactly.");
      if (!text(artifact.requirement_id) || !required.has(artifact.requirement_id)) errors.push("Artifact references an unknown requirement.");
      if (seenArtifacts.has(artifact.requirement_id)) errors.push(`Duplicate artifact requirement: ${artifact.requirement_id}.`);
      seenArtifacts.add(artifact.requirement_id);
      if (!nonempty(artifact.label)) errors.push("Artifact label is required.");
      if (!nonempty(artifact.url) || !validUrl(artifact.url)) errors.push("Artifact URL must be HTTP(S).");
      if (!text(artifact.sha256) || !SHA256.test(artifact.sha256)) errors.push("Artifact sha256 must have 64 lowercase hexadecimal characters.");
      if (!positiveInteger(artifact.bytes)) errors.push("Artifact bytes must be a positive integer.");
    }

    const combined = rows.filter(item => object(item) && item.id !== record.id).concat([record]);
    const byId = new Map(combined.map(item => [item.id, item]));
    const seenStages = new Set();
    for (const ref of list(record.upstream)) {
      if (!object(ref)) { errors.push("Each upstream reference must be an object."); continue; }
      if (!exactFields(ref, ["stage", "record_id"])) errors.push("Upstream fields do not match the schema exactly.");
      if (!dependencies(stage).includes(ref.stage)) errors.push(`Unexpected upstream stage: ${ref.stage}.`);
      if (seenStages.has(ref.stage)) errors.push(`Duplicate upstream stage: ${ref.stage}.`);
      seenStages.add(ref.stage);
      if (!identifier(ref.record_id)) errors.push("Upstream record_id must be a safe lowercase identifier.");
      if (ref.record_id === record.id) errors.push("A record cannot reference itself.");
      const parent = byId.get(ref.record_id);
      if (!parent) errors.push(`Missing upstream record: ${ref.record_id}.`);
      else {
        if (parent.stage !== ref.stage || parent.episode_id !== record.episode_id) errors.push(`Upstream ${ref.record_id} belongs to a different stage or episode.`);
        if (Date.parse(parent.created_at) > Date.parse(record.created_at)) errors.push(`Upstream ${ref.record_id} cannot be newer than its consumer.`);
        if (READY.includes(record.state) && (parent.kind !== "real_track1" || !READY.includes(parent.state))) errors.push(`Upstream ${ref.record_id} must be a checked or accepted real Track 1 result.`);
        if (READY.includes(record.state)) {
          for (const ancestor of list(parent.upstream)) {
            if (object(ancestor) && !list(record.upstream).some(binding => object(binding) && binding.stage === ancestor.stage && binding.record_id === ancestor.record_id)) errors.push(`Incoherent upstream chain: ${ref.record_id} consumed a different ${ancestor.stage} result.`);
          }
        }
      }
    }
    if (cycleIds(combined).has(record.id)) errors.push("Upstream references contain a cycle.");

    if (READY.includes(record.state)) {
      if (record.kind !== "real_track1") errors.push("Only real_track1 records may be checked or accepted.");
      if (!nonempty(record.delivery_url)) errors.push("Checked/accepted records require a delivery URL.");
      if (!COMMIT.test(record.source_commit || "")) errors.push("Checked/accepted records require the full source commit.");
      if (!nonempty(record.contributor)) errors.push("Checked/accepted records require an actual contributor.");
      if (!object(record.coverage) || !episode || record.coverage.frames !== episode.frames || record.coverage.objects_complete !== true) errors.push("Checked/accepted records require every original frame and all required objects.");
      const missing = missingRequirements(record, stage);
      if (missing.length) errors.push(`Incomplete required artifacts: ${missing.join(", ")}.`);
      const structural = object(record.checks) && record.checks.structural;
      if (!object(structural) || structural.status !== "pass" || !nonempty(structural.evidence_url)) errors.push("Checked/accepted records require structural pass evidence.");
      for (const id of dependencies(stage)) {
        if (!seenStages.has(id)) errors.push(`Required upstream stage is missing: ${id}.`);
      }
    }
    if (record.state === "accepted") {
      if (!nonempty(record.reviewer)) errors.push("Accepted records require an actual reviewer.");
      const quality = object(record.checks) && record.checks.quality;
      if (!object(quality) || quality.status !== "pass" || !nonempty(quality.evidence_url)) errors.push("Accepted records require quality pass evidence.");
      if (record.stage === "export" || record.stage === "baseline_export" || record.stage === "candidate_export" || (stage && stage.export_gate === true)) {
        const check = object(record.checks) && record.checks.export;
        if (!object(check) || check.status !== "pass" || !nonempty(check.evidence_url)) errors.push("Accepted export records require export pass evidence.");
      }
    }
    return [...new Set(errors)];
  }

  function validateBoard(board) {
    if (!object(board)) return ["Board must be an object."];
    const errors = [];
    if (board.schema_version !== "hoi_progress.v1") errors.push("Unsupported board schema_version.");
    if (!exactFields(board, ["schema_version", "updated_at", "source", "episodes", "stages", "records", "capabilities"])) errors.push("Board fields do not match the schema exactly.");
    if (!validTimestamp(board.updated_at)) errors.push("updated_at must be a valid ISO timestamp with timezone.");
    if (!exactFields(board.source, ["inventory_path", "repository", "revision", "scope"])) errors.push("source fields do not match the schema exactly.");
    else {
      if (board.source.inventory_path !== "docs/track1-videos/index.html#DATA" || board.source.repository !== "https://github.com/Seanwilliam2077/video-to-hoi") errors.push("Board source does not identify the project Track 1 inventory.");
      if (!text(board.source.revision) || !COMMIT.test(board.source.revision) || !nonempty(board.source.scope)) errors.push("Board source requires a full commit SHA and scope.");
    }
    for (const key of ["episodes", "stages", "records", "capabilities"]) {
      if (!Array.isArray(board[key])) errors.push(`${key} must be an array.`);
    }
    const seenEpisodes = new Set();
    for (const episode of list(board.episodes)) {
      if (!object(episode) || !Number.isSafeInteger(episode.ep) || episode.ep < 0 || episode.ep > 29 || episode.id !== `episode_${String(episode.ep).padStart(6, "0")}` || !positiveInteger(episode.frames)) { errors.push("Each episode requires its exact Track 1 id, ep 0–29 and positive integer frames."); continue; }
      if (seenEpisodes.has(episode.id)) errors.push(`Duplicate episode id: ${episode.id}.`);
      seenEpisodes.add(episode.id);
      if (!exactFields(episode, ["id", "ep", "object", "object_name", "frames", "seconds", "camera", "task", "poster_url", "video_url"]) || ["object", "object_name", "camera", "task"].some(key => !nonempty(episode[key])) || typeof episode.seconds !== "number" || !Number.isFinite(episode.seconds) || episode.seconds <= 0) errors.push(`Episode ${episode.id} metadata does not match the schema.`);
      if (!nonempty(episode.poster_url) || !validUrl(episode.poster_url) || !nonempty(episode.video_url) || !validUrl(episode.video_url) || !episode.video_url.includes("/track_1/videos/") || !episode.video_url.endsWith(`/${episode.id}.mp4`)) errors.push(`Episode ${episode.id} has an invalid Track 1 media reference.`);
    }
    if (seenEpisodes.size !== 30 || list(board.episodes).length !== 30) errors.push("Board requires exactly the 30 Track 1 episodes.");
    const seenStages = new Set();
    for (const stage of list(board.stages)) {
      if (!object(stage) || !identifier(stage.id)) { errors.push("Each stage requires a safe lowercase id."); continue; }
      if (seenStages.has(stage.id)) errors.push(`Duplicate stage id: ${stage.id}.`);
      seenStages.add(stage.id);
      if (!exactFields(stage, ["id", "module", "name", "label", "requirements", "dependencies"]) || !["platform", "human", "object", "physics"].includes(stage.module) || !nonempty(stage.name) || !nonempty(stage.label)) errors.push(`Stage ${stage.id} metadata does not match the schema.`);
      if (!Array.isArray(stage.dependencies) || !Array.isArray(stage.requirements)) errors.push(`Stage ${stage.id} requires dependencies and requirements arrays.`);
      if (new Set(dependencies(stage)).size !== dependencies(stage).length) errors.push(`Duplicate dependency for stage ${stage.id}.`);
      const required = list(stage.requirements).map(requirementId);
      if (list(stage.requirements).some(item => !exactFields(item, ["id", "label"]) || !nonempty(item.label))) errors.push(`Stage ${stage.id} requirement metadata does not match the schema.`);
      if (required.some(id => !identifier(id)) || new Set(required).size !== required.length) errors.push(`Stage ${stage.id} has invalid or duplicate requirements.`);
      const stageIds = Object.keys(REQUIREMENTS);
      if (!Object.hasOwn(REQUIREMENTS, stage.id) || stable(required) !== stable(REQUIREMENTS[stage.id]) || stable(dependencies(stage)) !== stable(stageIds.slice(0, stageIds.indexOf(stage.id)))) errors.push(`Stage ${stage.id} requirements or dependencies differ from the current profile.`);
    }
    if (seenStages.size !== 6 || list(board.stages).length !== 6) errors.push("Board requires exactly the six internal stage boundaries.");
    for (const stage of list(board.stages).filter(object)) {
      if (dependencies(stage).some(id => !seenStages.has(id) || id === stage.id)) errors.push(`Stage ${stage.id} has an unknown or self dependency.`);
    }
    const stageGraph = list(board.stages).filter(object).map(stage => ({id: stage.id, upstream: dependencies(stage).map(id => ({record_id: id}))}));
    if (cycleIds(stageGraph).size) errors.push("Stage dependency graph contains a cycle.");
    const seenCapabilities = new Set();
    for (const capability of list(board.capabilities)) {
      if (!exactFields(capability, ["module", "status", "summary", "evidence_url"]) || !["platform", "human", "object", "physics"].includes(capability.module) || !nonempty(capability.status) || !nonempty(capability.summary) || !nonempty(capability.evidence_url) || !validUrl(capability.evidence_url)) { errors.push("Capability metadata does not match the schema."); continue; }
      if (seenCapabilities.has(capability.module)) errors.push(`Duplicate module capability: ${capability.module}.`);
      seenCapabilities.add(capability.module);
    }
    const seen = new Set();
    for (const record of list(board.records)) {
      if (object(record) && seen.has(record.id)) errors.push(`Duplicate record id: ${record.id}.`);
      if (object(record)) seen.add(record.id);
      validateRecord(record, board, board.records).forEach(error => errors.push(`${object(record) ? record.id || "Record" : "Record"}: ${error}`));
    }
    return [...new Set(errors)];
  }

  function mergeRecords(existing, incoming) {
    if (!Array.isArray(existing) || !Array.isArray(incoming)) return {records: list(existing).slice(), errors: ["Existing and incoming records must be arrays."]};
    const errors = [];
    const merged = [];
    const byId = new Map();
    for (const record of existing.concat(incoming)) {
      if (!object(record) || !identifier(record.id)) { errors.push("Every merged record requires a safe lowercase id."); continue; }
      if (byId.has(record.id)) {
        if (stable(byId.get(record.id)) !== stable(record)) errors.push(`Conflicting immutable record id: ${record.id}.`);
      } else {
        byId.set(record.id, record);
      }
    }
    if (errors.length) return {records: existing.slice(), errors: [...new Set(errors)]};
    const known = new Set();
    for (const record of existing.concat(incoming)) {
      if (!known.has(record.id)) { merged.push(record); known.add(record.id); }
    }
    return {records: merged, errors: []};
  }

  function parseImport(input) {
    // JSON.parse alone silently discards duplicate keys. Scan valid JSON once
    // more to reject ambiguous producer declarations before any merge occurs.
    try {
      if (!text(input)) throw new Error("Import must contain JSON text.");
      const value = JSON.parse(input.replace(/^\uFEFF/, ""));
      const json = input.replace(/^\uFEFF/, "");
      let cursor = 0;
      const whitespace = () => { while (/\s/.test(json[cursor] || "") && cursor < json.length) cursor++; };
      function readString() {
        const start = cursor++;
        while (cursor < json.length) {
          const character = json[cursor++];
          if (character === "\\") cursor++;
          else if (character === '"') return JSON.parse(json.slice(start, cursor));
        }
        throw new Error("Unterminated JSON string.");
      }
      function scan(depth) {
        if (depth > 64) throw new Error("JSON nesting exceeds the import limit.");
        whitespace();
        const character = json[cursor];
        if (character === "{") {
          cursor++; whitespace();
          const keys = new Set();
          if (json[cursor] === "}") { cursor++; return; }
          while (cursor < json.length) {
            whitespace();
            const key = readString();
            if (keys.has(key)) throw new Error(`Duplicate JSON key: ${key}.`);
            keys.add(key);
            whitespace(); cursor++; // Colon; JSON.parse already checked syntax.
            scan(depth + 1); whitespace();
            if (json[cursor++] === "}") return;
          }
        } else if (character === "[") {
          cursor++; whitespace();
          if (json[cursor] === "]") { cursor++; return; }
          while (cursor < json.length) {
            scan(depth + 1); whitespace();
            if (json[cursor++] === "]") return;
          }
        } else if (character === '"') readString();
        else {
          while (cursor < json.length && !/[\s,\]}]/.test(json[cursor])) cursor++;
        }
      }
      scan(0);
      if (exactFields(value, ["schema_version", "records"])) {
        if (value.schema_version !== "hoi_progress.v1" || !Array.isArray(value.records)) throw new Error("Import envelope requires hoi_progress.v1 and a records array.");
        return {records: value.records, errors: []};
      }
      if (exactFields(value, RECORD_FIELDS)) return {records: [value], errors: []};
      throw new Error("Import must be one record or exactly {schema_version, records}; board catalogs and extra envelope fields are not accepted.");
    } catch (error) {
      return {records: [], errors: [String(error.message || error)]};
    }
  }

  return Object.freeze({STATES, KINDS, validateRecord, validateBoard, mergeRecords, latestFor, latestRecords, staleRecordIds, missingRequirements, parseImport, acceptedEpisodeIds});
});
