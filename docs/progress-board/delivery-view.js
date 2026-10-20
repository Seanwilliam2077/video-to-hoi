(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.HOIDelivery = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const HASH = /^[a-f0-9]{64}$/;
  const COMMIT = /^[a-f0-9]{40}$/;
  const EPISODE = /^episode_0000(?:[01][0-9]|2[0-9])$/;
  const PREVIEW = /^previews\/episode_0000(?:[01][0-9]|2[0-9])-(?:review|targets)\.jpg$/;
  const PROFILE = "sam3d_body_masks_camera.v1";
  const object = value => value !== null && typeof value === "object" && !Array.isArray(value);
  const text = value => typeof value === "string";
  const nonempty = value => text(value) && value.trim().length > 0;
  const list = value => Array.isArray(value) ? value : [];
  const positive = value => Number.isSafeInteger(value) && value > 0;
  const escape = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char]);
  const choose = (language, en, zh) => language === "en" ? en : zh;
  const stable = value => Array.isArray(value) ? "[" + value.map(stable).join(",") + "]" : object(value) ? "{" + Object.keys(value).sort().map(key => JSON.stringify(key) + ":" + stable(value[key])).join(",") + "}" : JSON.stringify(value);

  function safeUrl(value) {
    if (!nonempty(value) || /[\s\x00-\x1f\x7f\\]/.test(value)) return false;
    const authority = value.match(/^https?:\/\/([^/?#]+)/i);
    if (!authority || authority[1].includes("@")) return false;
    const host = authority[1].startsWith("[") ? authority[1].slice(1, authority[1].indexOf("]")) : authority[1].split(":")[0];
    if (!host || !/^[a-z0-9._:-]+$/i.test(host)) return false;
    if (!host.includes(":") && /^(?:[0-9]+|0x[0-9a-f]+)$/i.test(host.replace(/\.$/, "").split(".").at(-1))) {
      const parts = host.split(".");
      if (parts.length !== 4 || parts.some(part => !/^(?:0|[1-9][0-9]{0,2})$/.test(part) || Number(part) > 255)) return false;
    }
    try {
      const url = new URL(value);
      return ["http:", "https:"].includes(url.protocol) && Boolean(url.hostname) && !url.username && !url.password && url.port !== "0";
    } catch (_) { return false; }
  }

  function safePath(value) {
    return nonempty(value) && !/[\x00-\x1f\x7f\\?#:]/.test(value) && !value.startsWith("/") &&
      value.split("/").every((part, index, parts) => part === "" ? index === parts.length - 1 : part !== "." && part !== "..");
  }

  function recordErrors(record, stage = "inputs") {
    const errors = [];
    const fields = ["id", "episode_id", "stage", "iteration", "state", "kind", "created_at", "contributor", "reviewer", "summary", "delivery_url", "source_commit", "coverage", "upstream", "artifacts", "checks", "blocker", "next_action"];
    if (!object(record) || fields.some(key => !Object.hasOwn(record, key)) || Object.keys(record).some(key => !fields.includes(key))) return ["Record fields are invalid."];
    if (!nonempty(record.id) || !EPISODE.test(record.episode_id) || record.stage !== stage || record.kind !== "real_track1" || !COMMIT.test(record.source_commit)) errors.push("Record identity is invalid.");
    if (!["planned", "running", "blocked", "submitted", "checked", "accepted", "failed"].includes(record.state) ||
        !Number.isFinite(Date.parse(record.created_at)) || !safeUrl(record.delivery_url)) errors.push("Record state or URL is invalid.");
    for (const key of ["iteration", "contributor", "reviewer", "summary", "blocker", "next_action"]) if (!text(record[key])) errors.push(`Record ${key} is invalid.`);
    if (!object(record.coverage) || !positive(record.coverage.frames) || !(typeof record.coverage.objects_complete === "boolean" || (stage === "human" && record.coverage.objects_complete === null))) errors.push("Record coverage is invalid.");
    if (!Array.isArray(record.upstream) || record.upstream.some(item => !object(item) || !nonempty(item.stage) || !nonempty(item.record_id))) errors.push("Record upstream is invalid.");
    if (!Array.isArray(record.artifacts) || record.artifacts.some(item => !object(item) || !nonempty(item.requirement_id) || !text(item.label) || !safeUrl(item.url) || !HASH.test(item.sha256) || !positive(item.bytes))) errors.push("Record artifacts are invalid.");
    if (!object(record.checks) || ["structural", "quality", "export"].some(key => !object(record.checks[key]) || !["not_run", "pass", "fail"].includes(record.checks[key].status) ||
        !(record.checks[key].evidence_url === "" || safeUrl(record.checks[key].evidence_url)) ||
        (record.checks[key].status === "pass" && !safeUrl(record.checks[key].evidence_url)))) errors.push("Record checks are invalid.");
    return errors;
  }

  function entryErrors(entry) {
    if (!object(entry)) return ["Delivery entry must be an object."];
    const errors = recordErrors(entry.record);
    if (!EPISODE.test(entry.episode_id) || entry.episode_id !== entry.record?.episode_id || entry.profile !== PROFILE ||
        !positive(entry.frame_count) || entry.frame_count !== entry.record?.coverage?.frames) errors.push("Delivery identity or frame coverage is invalid.");
    for (const key of ["raw_verified", "body_inputs_verified", "native_readers_passed", "body_execution_completed"]) {
      if (typeof entry[key] !== "boolean") errors.push(`Delivery ${key} must be boolean.`);
    }
    if (entry.body_inputs_verified && !entry.raw_verified) errors.push("Body inputs cannot pass without verified raw inputs.");
    const camera = entry.camera;
    if (!object(camera) || !["fx", "fy", "cx", "cy"].every(key => typeof camera[key] === "number" && Number.isFinite(camera[key])) ||
        !(camera.fx > 0) || !(camera.fy > 0) || !positive(camera.width) || !positive(camera.height)) errors.push("Camera is invalid.");
    const review = entry.review;
    if (!object(review) || !Array.isArray(review.sample_frame_ids) || review.sample_frame_ids.some(frame => !Number.isSafeInteger(frame) || frame < 0 || frame >= entry.frame_count) ||
        new Set(review.sample_frame_ids).size !== review.sample_frame_ids.length || !nonempty(review.method) || !text(review.findings) || !nonempty(review.sampling_limitation)) errors.push("Review is invalid.");
    const archive = entry.archive;
    if (!object(archive) || !safeUrl(archive.url) || !positive(archive.bytes) || !HASH.test(archive.sha256) ||
        !safePath(archive.episode_path) || !archive.episode_path.split("/").includes(entry.episode_id)) errors.push("Archive is invalid.");
    if (!safeUrl(entry.source_video_url)) errors.push("Source video URL is invalid.");
    if (!Array.isArray(entry.evidence) || !entry.evidence.length || entry.evidence.some(item => !object(item) || !nonempty(item.id) || !nonempty(item.label) || !safeUrl(item.url) || !HASH.test(item.sha256) || !positive(item.bytes))) errors.push("Evidence is invalid.");
    if (!Array.isArray(entry.files) || !entry.files.length || entry.files.some(item => !object(item) || !safePath(item.path) || !nonempty(item.group) || !nonempty(item.kind) ||
        (!safeUrl(item.url) && !safePath(item.archive_path)) ||
        (Object.hasOwn(item, "url") && !safeUrl(item.url)) ||
        (Object.hasOwn(item, "archive_path") && !safePath(item.archive_path)) ||
        (Object.hasOwn(item, "sha256") && !HASH.test(item.sha256)) ||
        (Object.hasOwn(item, "bytes") && !positive(item.bytes)))) errors.push("File inventory is invalid.");
    if (!Array.isArray(entry.previews) || entry.previews.some(item => !object(item) || !text(item.url) || !PREVIEW.test(item.url) ||
        !item.url.startsWith(`previews/${entry.episode_id}-`) || !HASH.test(item.sha256) || !["mask_review", "target_review"].includes(item.kind) || !nonempty(item.label))) errors.push("Preview paths or metadata are invalid.");
    return errors;
  }

  function validateCatalog(catalog) {
    if (!object(catalog) || catalog.schema_version !== "hoi_delivery_catalog.v1" || !COMMIT.test(catalog.delivery_revision) || !Array.isArray(catalog.entries)) return ["Invalid delivery catalog envelope."];
    const errors = [];
    const ids = new Set();
    for (const entry of catalog.entries) {
      errors.push(...entryErrors(entry));
      if (ids.has(entry?.record?.id)) errors.push("Duplicate delivery record.");
      ids.add(entry?.record?.id);
    }
    return errors;
  }

  function entryFor(catalog, record) {
    if (!object(record) || validateCatalog(catalog).length) return null;
    const entry = catalog.entries.find(item => item.record.id === record.id);
    return entry && stable(entry.record) === stable(record) ? entry : null;
  }

  function link(url, label, className = "") {
    return safeUrl(url) ? `<a${className ? ` class="${escape(className)}"` : ""} href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)} ↗</a>` : `<span>${escape(label)}</span>`;
  }
  function bytes(value, language) {
    if (!positive(value)) return choose(language, "Size not registered", "未登记大小");
    const unit = value >= 1048576 ? `${(value / 1048576).toFixed(1)} MiB` : value >= 1024 ? `${(value / 1024).toFixed(1)} KiB` : `${value} B`;
    return `${unit} · ${value.toLocaleString(language === "en" ? "en-US" : "zh-CN")} bytes`;
  }
  function fingerprint(value, language) {
    return HASH.test(value) ? `<details class="delivery-hash"><summary>SHA256 · ${escape(value.slice(0, 12))}…</summary><code>${escape(value)}</code></details>` :
      `<span class="delivery-muted">${choose(language, "SHA256 not registered", "未登记 SHA256")}</span>`;
  }
  function fileCard(file, archive, language) {
    const size = file.kind === "directory" && !positive(file.bytes) ? choose(language, "Directory inside the archive; inspect individual file identities in the manifest.", "压缩包内目录；逐文件身份见清单。") : bytes(file.bytes, language);
    return `<li class="delivery-file"><div class="delivery-file-top"><span class="delivery-file-kind">${escape(file.kind)}</span><strong>${safeUrl(file.url) ? link(file.url, file.path) : escape(file.path)}</strong></div>
      <div class="delivery-file-meta">${escape(size)}</div>${file.sha256 ? fingerprint(file.sha256, language) : ""}
      ${file.archive_path ? `<p class="delivery-archive-path">${choose(language, "Inside ZIP", "ZIP 内路径")} <code>${escape(file.archive_path)}</code></p>${link(archive.url, choose(language, "Download archive to inspect", "下载压缩包查看"))}` : ""}</li>`;
  }
  function filesFor(entry, groups, language) {
    const files = entry.files.filter(file => groups.includes(file.group));
    return files.length ? `<ul class="delivery-files">${files.map(file => fileCard(file, entry.archive, language)).join("")}</ul>` : "";
  }
  function group(id, title, meta, body, open = false) {
    return `<details class="delivery-group" id="${id}"${open ? " open" : ""}><summary><span>${escape(title)}</span><span class="delivery-group-meta">${escape(meta)}</span></summary><div class="delivery-group-body">${body}</div></details>`;
  }
  function readiness(title, ready, language, pending) {
    return `<div class="delivery-readiness ${ready ? "delivery-ready" : "delivery-pending"}"><span>${escape(title)}</span><strong>${ready ? choose(language, "Verified", "已验证") : escape(pending || choose(language, "Pending", "待完成"))}</strong></div>`;
  }

  function renderInputs(entry, language = "zh") {
    if (entryErrors(entry).length) return `<section class="delivery-view"><p class="delivery-notice">${choose(language, "No matching verified delivery catalog is available for this exact record. Inspect the registered artifacts below.", "此记录暂无完全匹配的交付目录。请查看下方登记的产物；不据旧目录推断已准备完成。")}</p></section>`;
    const tr = (en, zh) => choose(language, en, zh);
    const count = entry.frame_count.toLocaleString(language === "en" ? "en-US" : "zh-CN");
    const statuses = `<div class="delivery-readiness-grid">
      ${readiness(tr("Original video + timeline", "原始视频与时间线"), entry.raw_verified, language)}
      ${readiness(tr("Body inputs", "Body 专用输入"), entry.body_inputs_verified, language)}
      ${readiness(tr("Object masks", "物体分割"), false, language, tr("Not in this package", "此包未交付"))}
      ${readiness(tr("Per-frame depth", "逐帧深度"), false, language, tr("Not in this package", "此包未交付"))}
      ${readiness(tr("Human scale alignment", "人体尺度对齐"), false, language, tr("Pending Human output", "等待 Human 结果"))}
      ${readiness(tr("Body execution", "Body 实际推理"), entry.body_execution_completed, language, tr("Not run", "尚未运行"))}</div>`;
    const checkSummary = ["structural", "quality", "export"].map(key => `${key}: ${entry.record.checks[key].status}`).join(" · ");
    const raw = group("delivery-raw", tr("Original video & frame timeline", "原始视频与帧时间线"), tr(`${count} original frames`, `${count} 原始帧`),
      `<p>${tr("All products retain original frame identities. Open the source video or inspect the timeline and source manifest.", "所有产物保留原始帧编号。可打开原始视频，或查看时间线与来源清单。")}</p><p>${link(entry.source_video_url, tr("Open original video", "打开原始视频"))}</p>${filesFor(entry, ["raw", "timeline"], language)}`);
    const body = group("delivery-body", tr("Human masks & observations", "人体 mask 与观察记录"), tr(`${count} frames`, `${count} 帧`),
      `<p>${tr("Per-frame target-human masks and observation records supply the Body input. Individual mask files are inside the downloadable ZIP.", "逐帧目标人体 mask 与观察记录提供 Body 输入。每帧 mask 文件可在压缩包内查看。")}</p>
       <p class="delivery-muted">${tr("Native reader validation", "原生读取器验证")}：${entry.native_readers_passed ? tr("passed", "通过") : tr("pending", "待完成")} · ${tr("Reading inputs does not establish reconstruction quality.", "读取通过不代表重建质量已验证。")}</p>${filesFor(entry, ["body", "masks", "observations"], language)}`);
    const cameraValues = ["fx", "fy", "cx", "cy", "width", "height"].map(key => `<div><dt>${key}</dt><dd>${escape(Number.isInteger(entry.camera[key]) ? entry.camera[key] : entry.camera[key].toFixed(3))}</dd></div>`).join("");
    const camera = group("delivery-camera", tr("Camera intrinsics", "相机内参"), `${entry.camera.width} × ${entry.camera.height}`,
      `<dl class="delivery-camera">${cameraValues}</dl><p class="delivery-muted">${tr("Estimated intrinsics in original image pixels. These are not ground-truth calibration or metric-scale evidence; sampled MoGe2 intrinsics do not constitute a per-frame depth delivery.", "在原始像素坐标下估计的内参，不是真实标定或米制尺度证据。采样提取 MoGe2 内参，不代表已交付逐帧深度。")}</p>${filesFor(entry, ["camera"], language)}`);
    const previews = entry.previews.map(preview => {
      const label = language === "en" ? preview.label : preview.kind === "mask_review" ? "人体 mask 全画面抽样复查" : "人体目标局部抽样复查";
      return `<figure class="delivery-preview"><a href="${escape(preview.url)}" target="_blank" rel="noopener noreferrer"><img src="${escape(preview.url)}" loading="lazy" decoding="async" alt="${escape(label)}"></a><figcaption>${escape(label)} · ${tr("Selected first sheet", "选取的首张拼图")} · <a href="${escape(preview.url)}" target="_blank" rel="noopener noreferrer">${tr("Open full image", "打开原图")} ↗</a></figcaption>${fingerprint(preview.sha256, language)}</figure>`;
    }).join("");
    const reviewMethod = entry.review.method === "machine_assisted_visual" ? tr("Machine-assisted sampled visual review", "机器辅助抽样可视复查") : entry.review.method;
    const qaArchivePath = `body-inputs/${entry.episode_id}/mask-qa/`;
    const review = group("delivery-review", tr("Actual mask review images & findings", "实际 mask 复查图与结论"), tr(`${entry.review.sample_frame_ids.length} sampled frames`, `${entry.review.sample_frame_ids.length} 个抽样帧`),
      `<p>${tr("Sampled original frame IDs", "抽样原始帧编号")}：<code>${escape(entry.review.sample_frame_ids.join(", "))}</code></p><p>${tr("Review method", "复查方式")}：${escape(reviewMethod)}</p>
      <p class="delivery-notice">${tr("The images below are selected first review sheets; they do not show every sampled frame listed above. The frame IDs describe the full review scope. Inspect the remaining QA sheets in the downloaded package.", "以下仅展示选取的首张复查拼图，并未展示上方全部抽样帧。上方帧编号表示完整复查范围；其余 QA 拼图请在下载包中查看。")}</p>
      <p class="delivery-muted">${tr("Sampled review does not verify every frame or boundary. Inspect findings and limitations in the original report below.", "抽样复查不代表已逐帧验证所有边界。请展开下方原始报告，查看复查发现与限制。")}</p>
      <p class="delivery-archive-path">${tr("All QA sheets inside ZIP", "ZIP 内全部 QA 拼图")}：<code>${escape(qaArchivePath)}</code> ${link(entry.archive.url, tr("Download package to inspect all sheets", "下载交付包查看全部拼图"))}</p>
      <div class="delivery-previews">${previews}</div>
      <details class="delivery-review-report"><summary>${tr("Review findings and limitations (original report)", "复查发现与限制（原始报告）")}</summary><div class="delivery-review-report-body"><h4>${tr("Findings", "复查发现")}</h4><p class="delivery-prose">${escape(entry.review.findings)}</p><h4>${tr("Sampling limitations", "抽样限制")}</h4><p class="delivery-prose">${escape(entry.review.sampling_limitation)}</p></div></details>
      ${filesFor(entry, ["qa"], language)}`);
    const inventory = `<p>${link(entry.archive.url, tr("Download all-30 Body input archive", "下载 30 视频 Body 输入包"), "button")}</p><p class="delivery-muted">${escape(bytes(entry.archive.bytes, language))}</p>
      ${fingerprint(entry.archive.sha256, language)}<p>${tr("This episode inside ZIP", "本视频在 ZIP 内的位置")}：<code>${escape(entry.archive.episode_path)}</code></p>
      <h4>${tr("Verification receipts", "验证回执")}</h4><ul class="delivery-files">${entry.evidence.map(item => `<li class="delivery-file">${link(item.url, item.label)}<div class="delivery-file-meta">${escape(bytes(item.bytes, language))}</div>${fingerprint(item.sha256, language)}</li>`).join("")}</ul>
      <details class="delivery-inventory"><summary>${tr("Browse every listed file", "浏览全部已列出的文件")} (${entry.files.length})</summary><ul class="delivery-files">${entry.files.map(file => fileCard(file, entry.archive, language)).join("")}</ul></details>`;
    const delivery = group("delivery-package", tr("Delivery package & verification receipts", "交付包与验证回执"), tr(`${entry.files.length} inventory entries`, `${entry.files.length} 项文件目录`), inventory);
    const objectSection = group("delivery-object", tr("Object segmentation", "物体分割"), tr("Missing from this Body package", "此 Body 交付包未包含"),
      `<p>${tr("Next delivery: each target object's per-frame masks, stable object identity, original frame mapping and segmentation checks. Human-only masks do not fulfill this requirement.", "下一项交付：每个目标物体的逐帧 mask、稳定物体 ID、原始帧映射与分割检查。人体 mask 不能代替物体分割。")}</p>`);
    const depth = group("delivery-depth", tr("Depth & scale alignment", "深度与尺度对齐"), tr("Pending", "待完成"),
      `<p>${tr("Inputs still needs per-frame depth and its provenance. Human scale alignment belongs to the Human output and requires the reconstructed human result. Intrinsics alone satisfy neither.", "Inputs 仍需逐帧深度及其来源记录。人体尺度对齐属于 Human 输出，依赖人体重建结果；仅有相机内参不能完成这两项。")}</p>`);
    return `<section class="delivery-view" aria-label="${tr("Body input delivery contents", "Body 输入交付内容")}">
      <div class="delivery-section-heading"><h3>${tr("Inspect the delivery", "查看实际交付")}</h3><code>${escape(entry.profile)}</code></div>${statuses}
      <p class="delivery-notice">${tr("Body profile verification and full Inputs acceptance are separate.", "Body 专用输入验证与完整 Inputs 验收分别统计。")} <strong>${escape(checkSummary)}</strong></p>
      ${raw}${body}${camera}${review}${delivery}${objectSection}${depth}</section>`;
  }

  const exact = (value, fields) => object(value) && Object.keys(value).length === fields.length && fields.every(key => Object.hasOwn(value, key));
  // Human previews are published result assets, never arbitrary remote embeds.
  const RELEASE = /^https:\/\/github\.com\/Seanwilliam2077\/video-to-hoi\/releases\/download\/([a-zA-Z0-9][a-zA-Z0-9._-]*)\/([a-zA-Z0-9][a-zA-Z0-9._-]*)$/;
  const HUMAN_PROFILE = "native_mhr_mediapipe_review.v1";
  const VIDEO_ROLES = ["baseline_full_body", "refined_full_body", "hands_comparison"];
  const IMAGE_ROLES = ["baseline_contact_sheet", "refined_contact_sheet", "hand_qa_frame"];
  const EVIDENCE_ROLES = ["checks", "depth_scale", "selection", "input_bindings"];
  function humanAsset(value) {
    return exact(value, ["url", "sha256", "bytes"]) && text(value.url) && RELEASE.test(value.url) && HASH.test(value.sha256) && positive(value.bytes);
  }
  function humanEntryErrors(entry) {
    const fields = ["record", "episode_id", "profile", "frame_count", "source_video_sha256", "mhr_model_sha256", "archive", "selected_candidate", "selection_reason", "quality_accepted", "shared_scene_scale_verified", "official_export_accepted", "evidence", "previews", "hand_guards"];
    if (!exact(entry, fields)) return ["Human preview fields are invalid."];
    const errors = recordErrors(entry.record, "human");
    const record = entry.record;
    if (entry.profile !== HUMAN_PROFILE || !EPISODE.test(entry.episode_id) || entry.episode_id !== record?.episode_id || !positive(entry.frame_count) || entry.frame_count !== record?.coverage?.frames || !HASH.test(entry.source_video_sha256) || !HASH.test(entry.mhr_model_sha256)) errors.push("Human preview identity or coverage is invalid.");
    if (record?.state !== "submitted" || record?.checks?.structural?.status !== "pass" || record?.checks?.quality?.status !== "not_run" || record?.checks?.export?.status !== "not_run" || ["quality_accepted", "shared_scene_scale_verified", "official_export_accepted"].some(key => entry[key] !== false)) errors.push("Human candidate preview cannot grant acceptance.");
    if (!["baseline", "refined"].includes(entry.selected_candidate) || !nonempty(entry.selection_reason)) errors.push("Human candidate selection is invalid.");
    const artifacts = list(record?.artifacts).filter(object);
    const human = artifacts.find(item => item.requirement_id === "human");
    const scale = artifacts.find(item => item.requirement_id === "depth_scale");
    if (artifacts.length !== 2 || !human || !scale || !humanAsset(entry.archive) || !["url", "sha256", "bytes"].every(key => human[key] === entry.archive[key])) errors.push("Human archive does not bind the registered output.");
    if (!exact(entry.evidence, EVIDENCE_ROLES) || EVIDENCE_ROLES.some(key => !humanAsset(entry.evidence[key]))) errors.push("Human evidence assets are invalid.");
    else if (!["url", "sha256", "bytes"].every(key => scale?.[key] === entry.evidence.depth_scale[key]) || record?.checks?.structural?.evidence_url !== entry.evidence.checks.url) errors.push("Human checks or scale evidence differs from the record.");
    const release = text(entry.archive?.url) ? entry.archive.url.match(RELEASE)?.[1] : null;
    const prefix = record?.id;
    const sameRelease = value => humanAsset(value) && value.url.match(RELEASE)[1] === release && value.url.match(RELEASE)[2].startsWith(`${prefix}-`);
    if (!/^[a-z0-9][a-z0-9._-]{0,95}$/.test(prefix) || !humanAsset(entry.archive) || entry.archive.url.match(RELEASE)[2] !== `${prefix}.zip`) errors.push("Human assets must name the exact immutable record.");
    if (!release || EVIDENCE_ROLES.some(key => !sameRelease(entry.evidence?.[key]))) errors.push("Human evidence must use the same immutable result release.");
    const roles = new Set();
    const urls = new Set();
    if (!Array.isArray(entry.previews)) errors.push("Human previews must be an array.");
    for (const preview of list(entry.previews)) {
      if (!exact(preview, ["role", "url", "sha256", "bytes", "content_type", "coverage"]) || ![...VIDEO_ROLES, ...IMAGE_ROLES].includes(preview.role) || roles.has(preview.role) || urls.has(preview.url)) { errors.push("Human preview roles are invalid or duplicated."); continue; }
      roles.add(preview.role); urls.add(preview.url);
      if (!sameRelease({url: preview.url, sha256: preview.sha256, bytes: preview.bytes})) errors.push("Human preview URL or identity is invalid.");
      const coverage = preview.coverage;
      if (VIDEO_ROLES.includes(preview.role)) {
        if (preview.content_type !== "video/mp4" || !text(preview.url) || !preview.url.endsWith(".mp4") || !exact(coverage, ["kind", "original_frame_count"]) || coverage.kind !== "full_sequence" || coverage.original_frame_count !== entry.frame_count) errors.push("Human video must cover the complete original sequence.");
      } else if (!["image/jpeg", "image/png"].includes(preview.content_type) || !text(preview.url) || !(preview.content_type === "image/jpeg" ? /\.jpe?g$/.test(preview.url) : preview.url.endsWith(".png")) || !exact(coverage, ["kind", "original_frame_ids"]) || coverage.kind !== "sampled" || !Array.isArray(coverage.original_frame_ids) || !coverage.original_frame_ids.length || coverage.original_frame_ids.some((id, index, ids) => !Number.isSafeInteger(id) || id < 0 || id >= entry.frame_count || (index > 0 && id <= ids[index - 1]))) errors.push("Human sample image coverage is invalid.");
    }
    if (VIDEO_ROLES.some(role => !roles.has(role)) || !IMAGE_ROLES.some(role => roles.has(role))) errors.push("Human preview needs three actual videos and at least one actual sample image.");
    if (!Array.isArray(entry.hand_guards) || entry.hand_guards.length !== 2 || entry.hand_guards.some((guard, index) => !exact(guard, ["side", "heldout_observed_frames", "sufficient", "before_px", "after_px", "nonregression"]) || guard.side !== ["left", "right"][index] || !(guard.heldout_observed_frames === null || (Number.isSafeInteger(guard.heldout_observed_frames) && guard.heldout_observed_frames >= 0 && guard.heldout_observed_frames <= entry.frame_count)) || guard.sufficient !== (guard.heldout_observed_frames >= 5) || ["before_px", "after_px"].some(key => !(guard[key] === null || (typeof guard[key] === "number" && Number.isFinite(guard[key]) && guard[key] >= 0))) || ![null, true, false].includes(guard.nonregression))) errors.push("Human held-out evidence summary is invalid.");
    if (entry.selected_candidate === "refined" && (!list(entry.hand_guards).some(guard => guard?.sufficient === true) || list(entry.hand_guards).some(guard => guard?.nonregression === false))) errors.push("Selected hand candidate failed the available local image guards.");
    return errors;
  }
  function validateHumanCatalog(catalog) {
    if (!exact(catalog, ["schema_version", "entries"]) || catalog.schema_version !== "hoi_human_catalog.v1" || !Array.isArray(catalog.entries)) return ["Invalid Human catalog envelope."];
    const errors = [], ids = new Set();
    for (const entry of catalog.entries) {
      errors.push(...humanEntryErrors(entry));
      if (ids.has(entry?.record?.id)) errors.push("Duplicate Human preview record.");
      ids.add(entry?.record?.id);
    }
    return errors;
  }
  function humanEntryFor(catalog, record) {
    if (!object(record) || validateHumanCatalog(catalog).length) return null;
    const entry = catalog.entries.find(item => item.record.id === record.id);
    return entry && stable(entry.record) === stable(record) ? entry : null;
  }
  function renderHuman(entry, language = "zh") {
    if (humanEntryErrors(entry).length) return "";
    const tr = (en, zh) => choose(language, en, zh);
    const abstained = entry.hand_guards.every(guard => guard.heldout_observed_frames === null);
    const labels = {
      baseline_full_body: tr("Fixed identity and skeleton baseline", "固定身份与骨架的基线"),
      refined_full_body: abstained ? tr("Hand fitting abstained: unchanged baseline", "手部拟合已暂缓：保持原基线") : tr("MediaPipe hand fitting candidate", "MediaPipe 手部拟合候选"),
      hands_comparison: tr("Hand close-ups: source, observations and both candidates", "手部近景：原视频、观测与两个候选"),
      baseline_contact_sheet: tr("Baseline sampled frames", "基线抽样画面"),
      refined_contact_sheet: tr("Hand candidate sampled frames", "手部候选抽样画面"),
      hand_qa_frame: tr("Actual hand comparison sample", "真实手部对照抽样画面")
    };
    const selected = entry.selected_candidate === "baseline" ? tr("Fixed-body baseline", "固定身体基线") : tr("Hand fitting candidate", "手部拟合候选");
    const media = entry.previews.map(item => {
      const video = VIDEO_ROLES.includes(item.role);
      const coverage = video ? tr(`${entry.frame_count} original frames`, `完整 ${entry.frame_count} 原始帧`) : tr(`Sampled original frame IDs: ${item.coverage.original_frame_ids.join(", ")}`, `抽样原始帧编号：${item.coverage.original_frame_ids.join("，")}`);
      return `<figure class="delivery-preview delivery-human-media" data-human-role="${escape(item.role)}"><figcaption><h4>${escape(labels[item.role])}</h4><p>${escape(coverage)}</p></figcaption>${video ? `<video controls playsinline preload="none" src="${escape(item.url)}" aria-label="${escape(labels[item.role])}"></video>` : `<a href="${escape(item.url)}" target="_blank" rel="noopener noreferrer"><img loading="lazy" decoding="async" src="${escape(item.url)}" alt="${escape(labels[item.role])}"></a>`}<p>${link(item.url, tr("Open or download this media", "打开或下载此媒体"))} · ${escape(bytes(item.bytes, language))}</p>${fingerprint(item.sha256, language)}</figure>`;
    }).join("");
    const handRows = entry.hand_guards.map(guard => `<tr><td>${escape(guard.side === "left" ? tr("Left hand", "左手") : tr("Right hand", "右手"))}</td><td>${guard.heldout_observed_frames ?? tr("Unknown", "未知")}</td><td>${guard.before_px === null ? "—" : guard.before_px.toFixed(2)} → ${guard.after_px === null ? "—" : guard.after_px.toFixed(2)} px</td><td>${escape(guard.sufficient ? tr("Sufficient for the local guard", "满足局部检查的数量要求") : tr("Insufficient; keep uncertainty", "证据不足，保留不确定性"))}</td></tr>`).join("");
    const evidenceLabels = {checks: tr("Native and hand checks", "原生参数与手部检查"), depth_scale: tr("Depth and scale limitations", "深度与尺度限制"), selection: tr("Exact local candidate selection", "准确的本地候选选择记录"), input_bindings: tr("Source and runtime identities", "输入来源与运行环境身份")};
    return `<section class="delivery-view delivery-human" aria-label="${tr("Published Human candidate previews", "已发布人体候选预览")}"><div class="delivery-section-heading"><h3>${tr("Inspect the actual Human results", "查看真实人体重建结果")}</h3><code>${escape(entry.profile)}</code></div><p><strong>${tr("Selected local candidate", "当前本地选择")}：${escape(selected)}</strong></p><details class="delivery-review-report"><summary>${tr("Selection rationale (original report)", "选择依据（原始报告）")}</summary><p class="delivery-review-report-body">${escape(entry.selection_reason)}</p></details><p class="delivery-notice">${tr("These are submitted candidates, not accepted reconstructions. Native structure has been checked; reconstruction quality, true 3D hand contact, metric/shared scene scale and official export remain pending. Media previews do not change dashboard status. No hidden-reference score is available.", "这些是已递交候选，尚未通过重建验收。原生参数结构已检查；重建质量、真实三维手部接触、米制与人–物共同尺度、正式导出仍待验证。媒体预览不会改变看板状态，当前没有隐藏参考答案得分。")}</p><div class="delivery-human-grid">${media}</div><div class="delivery-human-table"><table><thead><tr><th>${tr("Hand", "手部")}</th><th>${tr("Held-out observed frames", "留出观测帧")}</th><th>${tr("Image agreement before → after", "图像偏差：拟合前 → 后")}</th><th>${tr("Evidence scope", "证据范围")}</th></tr></thead><tbody>${handRows}</tbody></table></div><p class="delivery-muted">${tr("Image agreement with MediaPipe observations does not measure hidden 3D accuracy. Sample images cover only the listed frames. Compare grasp, occlusion and release in the full videos.", "与 MediaPipe 观测的图像偏差不等于隐藏三维真值误差。抽样图片只覆盖列出的帧；请结合完整视频检查抓取、遮挡和松手。")}</p><ul class="delivery-files">${EVIDENCE_ROLES.map(role => `<li class="delivery-file">${link(entry.evidence[role].url, evidenceLabels[role])}${fingerprint(entry.evidence[role].sha256, language)}</li>`).join("")}</ul></section>`;
  }

  const REQUIREMENTS = {
    inputs: [["camera", "Camera and source frame identity", "相机与原始帧身份"], ["masks", "Per-frame segmentation masks", "逐帧分割 mask"], ["depth", "Per-frame depth evidence", "逐帧深度证据"]],
    human: [["human", "Human trajectory and representation", "人体轨迹与表示契约"], ["depth_scale", "Human depth-scale alignment", "人体深度与尺度对齐"]],
    objects: [["object_record", "Object identity, frame and scale", "物体身份、坐标系与尺度"], ["mesh", "Object mesh and provenance", "物体网格与来源"]],
    motion: [["transforms", "Full-frame object transforms", "全帧物体位姿"], ["confidence", "Per-frame confidence", "逐帧置信度"]],
    refine: [["refined_human", "Refined human trajectory", "优化后人体轨迹"], ["refined_objects", "Refined object trajectory", "优化后物体轨迹"]],
    export: [["metadata", "Episode, object and frame manifest", "视频、物体与帧清单"], ["parquet", "Complete trajectory tables", "完整轨迹表"], ["mhr_arrays", "Human parameter arrays", "人体参数数组"], ["meshes", "Object meshes and pose pairing", "物体网格与位姿配对"]]
  };
  const NEXT = {
    inputs: ["Complete object masks and depth; the independently verified Body input profile can be handed over separately.", "补齐物体分割与深度；独立验证的 Body 输入可以单独交接。"],
    human: ["Prepare the receiving Body environment and weights, run reconstruction, then register trajectories and scale evidence.", "准备接收端 Body 环境与权重，执行人体重建，再登记人体轨迹和尺度证据。"],
    objects: ["Reconstruct target objects, establish metric scale and record mesh provenance.", "重建目标物体，确认米制尺度并记录网格来源。"],
    motion: ["Track objects on the fixed original timeline and register transforms with confidence.", "在固定原始时间线上跟踪物体，登记位姿与置信度。"],
    refine: ["Evaluate and register refined human and object trajectories with comparable evidence.", "评估并登记优化后的人体与物体轨迹，提供可比对证据。"],
    export: ["Package validated trajectories, human parameters and paired meshes; run the official submission checks.", "打包已验证轨迹、人体参数与配对网格，执行正式提交检查。"]
  };

  function renderArtifacts(record, stage, language = "zh") {
    const tr = (en, zh) => choose(language, en, zh);
    const id = text(stage) ? stage : stage?.id;
    const defaults = REQUIREMENTS[id] || [];
    const requirements = object(stage) && Array.isArray(stage.requirements) ? stage.requirements.map(item => object(item) ? item : {id: item, label: item}) : defaults.map(item => ({id: item[0], label: tr(item[1], item[2])}));
    const artifacts = list(record?.artifacts).filter(object);
    const complete = artifact => safeUrl(artifact.url) && HASH.test(artifact.sha256) && positive(artifact.bytes);
    const requirementLabel = requirement => {
      const known = defaults.find(item => item[0] === requirement.id);
      return known ? tr(known[1], known[2]) : requirement.label || requirement.id;
    };
    const card = (artifact, requirement) => {
      const ready = artifact && complete(artifact);
      return `<article class="delivery-artifact ${ready ? "delivery-artifact-present" : "delivery-artifact-missing"}">
        <div class="delivery-artifact-top"><h4>${escape(requirementLabel(requirement))}</h4><span class="delivery-state">${ready ? tr("Registered", "已登记") : artifact ? tr("Incomplete registration", "登记不完整") : tr("Not delivered", "未交付")}</span></div>
        <code>${escape(requirement.id)}</code>${artifact ? `<p>${escape(artifact.label || requirementLabel(requirement))}</p><p>${safeUrl(artifact.url) ? link(artifact.url, tr("Open submitted content", "打开递交内容"), "button") : tr("No valid content link registered.", "尚无有效递交链接。")}</p><div class="delivery-file-meta">${escape(bytes(artifact.bytes, language))}</div>${fingerprint(artifact.sha256, language)}` : `<p class="delivery-muted">${tr("The output content and verification evidence have not been registered.", "尚未登记输出内容和验证证据。")}</p>`}</article>`;
    };
    const cards = requirements.flatMap(requirement => {
      const items = artifacts.filter(artifact => artifact.requirement_id === requirement.id);
      return items.length ? items.map(artifact => card(artifact, requirement)) : [card(null, requirement)];
    });
    for (const artifact of artifacts.filter(item => !requirements.some(requirement => requirement.id === item.requirement_id))) cards.push(card(artifact, {id: artifact.requirement_id, label: artifact.label || artifact.requirement_id}));
    const next = record?.next_action || (NEXT[id] ? tr(...NEXT[id]) : tr("Register this stage's output and its verification evidence.", "登记此阶段输出及其验证证据。"));
    return `<section class="delivery-view delivery-artifacts" aria-label="${tr("Stage delivery inventory", "阶段交付清单")}"><div class="delivery-section-heading"><h3>${tr("Stage deliverables", "阶段交付内容")}</h3><span class="delivery-muted">${artifacts.filter(complete).length} ${tr("registered artifacts", "项已登记产物")}</span></div>
      <p class="delivery-muted">${tr("Registered links, sizes and hashes describe a delivery declaration; acceptance still depends on the stage checks.", "链接、大小和哈希表示交付登记；是否验收通过仍取决于阶段检查。")}</p>
      ${record && safeUrl(record.delivery_url) ? `<p>${link(record.delivery_url, tr("Open delivery folder", "打开交付目录"), "button")}</p>` : ""}
      <div class="delivery-artifact-grid">${cards.join("")}</div><p class="delivery-next"><strong>${tr("Next step", "下一步")}：</strong>${escape(next)}</p></section>`;
  }

  return Object.freeze({validateCatalog, entryFor, renderInputs, renderArtifacts, validateHumanCatalog, humanEntryFor, renderHuman});
});
