/* Browser-local drafts; the immutable Git registry remains the shared source. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const repo = 'https://github.com/Seanwilliam2077/video-to-hoi';
  const raw = 'https://raw.githubusercontent.com/Seanwilliam2077/video-to-hoi/main/tracking/board.json';
  const draftKey = 'hoi-progress-drafts-v1';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const link = (url, label) => `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)} ↗</a>`;
  const zh = {
    boardNav:'视频交付看板',attentionNav:'需要跟进',capabilitiesNav:'模块准备情况',workingAreas:'工作分区',openHub:'打开团队 Hub ↗',deliveryGuide:'交付目录与模板 ↗',floatingRoles:'按任务协作，角色自由流动',title:'30 段视频，一张交付地图。',subtitle:'从输入到最终导出，追踪每次交付、缺失材料和下一步。',howTo:'如何更新',refresh:'↻ 刷新仓库',export:'导出草稿',videos:'跟踪视频',videoGroups:'10 组物体',fullSet:'完整 Track 1 清单',registered:'已登记真实阶段',pendingReview:'等待检查',stageUnits:'阶段',reviewHint:'真实交付已递交，尚未通过结构检查',acceptedVideos:'已登记验收的视频',acceptedHint:'六个阶段及最终导出均已登记验收',checkScope:'检查范围 ↗',videoMatrix:'视频 × 交付阶段',matrix:'矩阵',cards:'卡片',iteration:'迭代',clearFilters:'重置筛选',noResults:'没有符合条件的视频',tryFilter:'试试其他筛选条件，或清空搜索。',cellHint:'点击任一阶段，查看交付清单与迭代历史。',import:'导入记录 JSON',sharedPreparation:'共享基础准备',separateProgress:'工程准备与逐视频结果分别记录',footerNote:'状态由交付记录申报；文件真实性与模型质量以所附检查证据为准。',search:'搜索视频编号、物体或任务…',allObjects:'全部物体',allVideos:'全部 30 个视频',baseline:'基线 · 12 / 16',devmini:'Dev mini · 5 个视频',allStages:'全部阶段',allStates:'全部状态',allIterations:'所有迭代 · 各阶段最新',unreported:'未登记',planned:'已计划',running:'进行中',submitted:'已递交',checked:'检查通过',accepted:'已验收',blocked:'受阻',failed:'失败',stale:'需重新检查',inputs:'输入证据',human:'人体重建',objects:'物体重建',motion:'物体运动',refine:'交互优化',stageExport:'最终导出',platformModule:'M1 · 平台与导出',humanModule:'M2 · 人体重建',objectModule:'M3 · 物体与运动',physicsModule:'M4 · 交互优化',real_track1:'真实 Track 1',synthetic:'合成数据',engineering:'工程准备',frames:'帧',seconds:'秒',registeredShort:'登记',acceptedShort:'验收',local:'本机草稿',sourceSnapshot:'Git 快照',sourceRemote:'仓库最新快照',contextEmpty:'尚未登记逐视频交付包。当前 30 个视频均显示“未登记”，并不代表此前没有实验。共享工程准备单独列在下方。',contextActive:'状态基于当前迭代的交付声明；本机草稿尚未同步到 Git。更换上游版本会标记依赖结果需重新检查。',contextDraft:' 本机草稿已叠加到预览统计；提交 Git 后才成为团队记录。',resultUnit:'个视频',noRecord:'这一阶段尚无交付登记。可以先记录任务与下一步，再逐步补齐文件和检查报告。',requirements:'交付清单',missing:'缺少登记',artifact:'文件',evidence:'检查证据',structural:'结构检查',quality:'质量评审',exportCheck:'导出检查',not_run:'未检查',pass:'通过',fail:'未通过',contributor:'贡献者',reviewer:'评审者',coverage:'帧覆盖',objectCoverage:'物体完整',yes:'已声明完整',no:'未完整',unknown:'未知',commit:'代码版本',upstream:'固定上游版本',noUpstream:'此阶段没有上游依赖。',unbound:'未绑定',history:'迭代历史',newRecord:'登记下一次进展',removeDraft:'移除本机草稿',viewRecord:'查看',deliveryRecord:'模块递交记录',sourceVideo:'查看视频来源',moduleLocation:'递交到模块目录',nextAction:'下一步',blocker:'阻碍',staleNotice:'引用的上游已更新或已过期。保留历史记录；核对新版本后创建新的交付记录。',checksNote:'此处显示报告声明。登记校验不读取文件内容，也不替代质量评审。',recordId:'记录 ID',summary:'本次进展',state:'状态',kind:'证据类型',deliveryUrl:'模块交付记录 URL',sourceCommit:'源代码 Git SHA（40 位）',framesCovered:'覆盖帧数',objectsComplete:'全部物体已覆盖？',artifactHint:'尚未提供的产物留空。填写的每项必须同时提供 URL、SHA-256 和文件字节数。',url:'文件 URL',hash:'SHA-256（64 位）',bytes:'字节数',checkUrl:'报告 URL',upstreamHint:'显式选择实际使用的版本；允许复用另一迭代的基线。不要绑定未实际使用的结果。',saveDraft:'保存本机草稿',cancel:'取消',draftHint:'这会创建一条新的本机记录，不覆盖历史。导出并提交到 Git 后，团队才能看到。',saved:'已保存本机草稿；尚未提交到 Git。',removed:'本机草稿已移除。',noDrafts:'没有待导出的本机草稿。',exported:'草稿已导出。按“如何更新”中的命令导入仓库。',imported:'记录已导入本机草稿；尚未提交到 Git。',refreshed:'已更新仓库快照。',refreshing:'正在读取仓库…',storageError:'浏览器无法保存草稿，请先导出 JSON 备份。',loadError:'无法加载看板数据。请通过本地 HTTP 服务打开，或点击刷新仓库重试。',importError:'无法导入记录：',validationError:'请补齐以下信息：',snapshot:'快照生成',basis:'代码基准',guideIntro:'每个人在自己的环境运行模型，通过 Git 共享交付索引和检查证据。大文件留在结果发布区或约定存储中。',guide1:'完成产物并运行检查',guide1Body:'按所属模块的 deliveries 模板记录实际输出、校验报告、代码版本和固定上游记录 ID。这里的六阶段是当前内部边界，原生模型适配与正式验收仍需继续接入。',guide2:'在看板准备进展草稿',guide2Body:'点开视频阶段，填写贡献者、迭代和本次进展。产物项需要 URL、SHA-256、字节数。已验收还需要评审者与质量报告。草稿只存在于当前浏览器。',guide3:'导出并提交 Git',guide3Body:'导出草稿 JSON 后，在仓库运行以下命令，再通过任务分支提交 PR。合并后，各成员点击“刷新仓库”读取最新进展。',guide4:'新迭代保留历史',guide4Body:'状态变化或重新运行都创建新记录 ID。绑定准确的上游版本；上游更新会提示下游重新检查，其他独立迭代不会覆盖基线。',privateNote:'在线页面默认仅站点所有者可访问；团队可从仓库启动同一页面。浏览器草稿不是多人同步数据库。',localPreview:'本地预览',metadataTask:'任务描述（数据集原文）',historyEmpty:'还没有历史记录。',currentRecord:'当前记录',historical:'历史版本',allViewHint:'筛选“所有迭代”时，各阶段可能来自不同实验；整条视频验收按最终导出绑定的完整上游链计算。',attentionEmpty:'没有已登记的阻塞、失败、待查或过期结果。',present:'已登记',invalidStorage:'本机保存的草稿无法通过验证，已保留原数据但没有载入。请导出恢复副本后处理。',recover:'导出恢复副本',cancelEdit:'取消编辑',none:'无',viewLatest:'返回当前阶段',saving:'正在保存…'
  };
  const en = {
    boardNav:'Delivery board',attentionNav:'Needs attention',capabilitiesNav:'Module readiness',workingAreas:'WORKING AREAS',openHub:'Open team Hub ↗',deliveryGuide:'Delivery folders & templates ↗',floatingRoles:'Flexible roles, shared ownership',title:'30 videos. Every handoff in view.',subtitle:'Track deliverables, missing evidence and the next step — from input to final export.',howTo:'How to update',refresh:'↻ Refresh repository',export:'Export drafts',videos:'VIDEOS TRACKED',videoGroups:'10 object groups',fullSet:'Complete Track 1 inventory',registered:'REAL STAGES REGISTERED',pendingReview:'AWAITING CHECKS',stageUnits:'stages',reviewHint:'Real deliveries submitted, not structurally checked',acceptedVideos:'VIDEOS DECLARED ACCEPTED',acceptedHint:'All six stages and final export declared accepted',checkScope:'Check scope ↗',videoMatrix:'Videos × delivery stages',matrix:'Matrix',cards:'Cards',iteration:'Iteration',clearFilters:'Reset filters',noResults:'No matching videos',tryFilter:'Try another filter or clear your search.',cellHint:'Open any stage for its checklist and iteration history.',import:'Import record JSON',sharedPreparation:'Shared foundations',separateProgress:'Engineering readiness is separate from per-video results',footerNote:'Statuses are declarations; artifact integrity and model quality require the linked evidence.',search:'Search video, object or task…',allObjects:'All objects',allVideos:'All 30 videos',baseline:'Baseline · 12 / 16',devmini:'Dev mini · 5 videos',allStages:'All stages',allStates:'All statuses',allIterations:'All iterations · latest by stage',unreported:'Unreported',planned:'Planned',running:'Running',submitted:'Submitted',checked:'Checked',accepted:'Accepted',blocked:'Blocked',failed:'Failed',stale:'Recheck',inputs:'Inputs',human:'Human',objects:'Objects',motion:'Motion',refine:'Refinement',stageExport:'Export',platformModule:'M1 · Platform & export',humanModule:'M2 · Human reconstruction',objectModule:'M3 · Objects & motion',physicsModule:'M4 · Interaction refinement',real_track1:'Real Track 1',synthetic:'Synthetic',engineering:'Engineering',frames:'frames',seconds:'s',registeredShort:'registered',acceptedShort:'accepted',local:'Local draft',sourceSnapshot:'Git snapshot',sourceRemote:'Repository snapshot',contextEmpty:'No per-video delivery packages are registered yet. All 30 videos show “Unreported”; earlier experiments may exist. Shared engineering foundations are listed separately below.',contextActive:'Statuses follow the selected iteration’s declarations. Local drafts have not been committed to Git. Updated upstream records flag dependent results for rechecking.',contextDraft:' Local drafts are included in these preview totals; they become team records only after a Git submission.',resultUnit:'videos',noRecord:'No delivery is registered for this stage. Start with a task and next action, then add outputs and check reports.',requirements:'Delivery checklist',missing:'Not registered',artifact:'Artifact',evidence:'Check evidence',structural:'Structure',quality:'Quality',exportCheck:'Export',not_run:'Not run',pass:'Pass',fail:'Fail',contributor:'Contributor',reviewer:'Reviewer',coverage:'Frame coverage',objectCoverage:'All objects',yes:'Declared complete',no:'Incomplete',unknown:'Unknown',commit:'Code revision',upstream:'Pinned upstream records',noUpstream:'This stage has no upstream dependencies.',unbound:'Not bound',history:'Iteration history',newRecord:'Register next progress',removeDraft:'Remove local draft',viewRecord:'View',deliveryRecord:'Module delivery record',sourceVideo:'View video source',moduleLocation:'Module delivery folder',nextAction:'Next action',blocker:'Blocker',staleNotice:'An upstream record has changed or is stale. Keep this history; verify the new inputs and create a new delivery record.',checksNote:'Report declarations only. Registry validation does not inspect artifact bytes or replace quality review.',recordId:'Record ID',summary:'Progress summary',state:'Status',kind:'Evidence type',deliveryUrl:'Module delivery record URL',sourceCommit:'Source Git SHA (40 characters)',framesCovered:'Frames covered',objectsComplete:'All objects covered?',artifactHint:'Leave missing outputs blank. Each provided output needs a URL, SHA-256 hash and positive byte count.',url:'Artifact URL',hash:'SHA-256 (64 characters)',bytes:'Bytes',checkUrl:'Report URL',upstreamHint:'Select the exact records actually used. Explicit reuse from another iteration is allowed.',saveDraft:'Save local draft',cancel:'Cancel',draftHint:'Creates a new local record and preserves history. Export and commit to Git to share with the team.',saved:'Saved locally; not yet submitted to Git.',removed:'Local draft removed.',noDrafts:'No local drafts to export.',exported:'Drafts exported. Follow “How to update” to import them into Git.',imported:'Imported as local drafts; not yet submitted to Git.',refreshed:'Repository snapshot updated.',refreshing:'Reading repository…',storageError:'Browser storage is unavailable. Export a JSON backup now.',loadError:'Could not load the board. Open it through a local HTTP server, or try Refresh repository.',importError:'Record import failed: ',validationError:'Please resolve:',snapshot:'Snapshot built',basis:'Code basis',guideIntro:'Run models on your own machine. Share delivery indexes and check evidence through Git; keep large outputs in result releases or agreed storage.',guide1:'Produce outputs and run checks',guide1Body:'Use the module’s deliveries template for actual artifacts, check reports, code revision and pinned upstream IDs. These six stages are internal boundaries; native adapters and official acceptance still need integration.',guide2:'Prepare a progress draft',guide2Body:'Open a stage and enter the contributor, iteration and summary. Artifact entries need a URL, SHA-256 and byte count. Acceptance also requires a reviewer and quality report. Drafts live only in this browser.',guide3:'Export and submit through Git',guide3Body:'Export draft JSON, run the commands below from the repository, then submit a task-branch PR. After merging, teammates can refresh the repository snapshot.',guide4:'Preserve iteration history',guide4Body:'Create a new record ID for status changes or reruns. Pin exact upstream records. Updated inputs flag descendants for rechecking without overwriting separate experiments.',privateNote:'The hosted site is owner-private by default. Teammates can run the same page from Git. Browser drafts are not a shared database.',localPreview:'Local preview',metadataTask:'Task description (dataset text)',historyEmpty:'No history yet.',currentRecord:'Current record',historical:'Historical record',allViewHint:'“All iterations” may combine stages from different experiments; complete-video acceptance follows the final export’s exact upstream chain.',attentionEmpty:'No registered blocked, failed, pending or stale results.',present:'Registered',invalidStorage:'Saved local drafts failed validation. The original data is retained but not loaded. Export a recovery copy before resolving it.',recover:'Export recovery copy',cancelEdit:'Cancel editing',none:'None',viewLatest:'Back to current stage',saving:'Saving…'
  };
  Object.assign(zh, {
    title: 'Track 1 视频交付看板',
    subtitle: '跟踪全部 30 个视频的六个阶段、交付材料与迭代记录。',
    workingAreas: '递交分区', repository: '仓库', teamHub: '团队主页',
    projectStatus: '项目进展', reference: '比赛资料', platformNav: '① 平台',
    humanNav: '② 人体', objectNav: '③ 物体', physicsNav: '④ 物理'
  });
  Object.assign(en, {
    title: 'Track 1 delivery board',
    subtitle: 'Track all 30 videos across six stages, delivery evidence and iterations.',
    workingAreas: 'Delivery folders', repository: 'Repository', teamHub: 'Team hub',
    projectStatus: 'Status', reference: 'Reference', platformNav: '① Platform',
    humanNav: '② Human', objectNav: '③ Object', physicsNav: '④ Physics',
    videos: 'Videos tracked', registered: 'Real stages registered',
    pendingReview: 'Awaiting checks', acceptedVideos: 'Videos declared accepted'
  });
  const objectZh = {hula_hoop:'呼啦圈',big_red_bowl:'大红碗',iron:'熨斗',white_desk:'白色桌子',black_pan:'黑色平底锅',foam_grass_block:'草坪泡沫块',paint_roller:'油漆滚筒',pink_foam_roll:'粉色泡沫轴',short_wood_stool:'矮木凳',white_laptop_cart:'白色电脑推车'};
  const requirementZh = {camera:'相机与源帧身份',masks:'逐帧分割掩码',depth:'深度与尺度证据',human:'人体轨迹与表示契约',depth_scale:'深度尺度记录',object_record:'物体身份、坐标系与尺度',mesh:'物体网格与来源',transforms:'全帧物体变换',confidence:'逐帧置信度',refined_human:'优化后人体轨迹',refined_objects:'优化后物体轨迹',metadata:'视频、物体与帧清单',parquet:'完整轨迹数据表',mhr_arrays:'人体参数数组与表示契约',meshes:'物体网格与姿态配对'};
  const capabilityZh = {platform:'已有内部交付门禁与目录。原生产物导入、跨机器包迁移和正式导出验收仍待接入。',human:'已有内部人体数组结构检查。真实原生人体适配器与逐视频验收结果尚未登记。',object:'已有网格和运动结构检查。真实重建、跟踪和逐视频验收结果尚未登记。',physics:'已有优化输出结构检查。真实优化、交互质量和逐视频验收结果尚未登记。'};
  let language = 'zh', board = null, drafts = [], records = [], stale = new Set(), latest = new Map();
  let remote = false, attention = false, view = matchMedia('(max-width:760px)').matches ? 'cards' : 'matrix';
  let selectedEpisode = null, selectedStage = null, selectedRecordId = null, editing = false, recovery = null;
  let toastTimer;
  const t = key => (language === 'zh' ? zh : en)[key] || key;
  const stageName = id => t(id === 'export' ? 'stageExport' : id);
  const objectName = ep => language === 'zh' ? objectZh[ep.object] || ep.object_name : ep.object_name;
  const reqName = req => language === 'zh' ? requirementZh[req.id] || req.label : req.label;
  const iteration = () => $('iteration-filter').value || undefined;
  const draftIds = () => new Set(drafts.map(r => r.id));
  const stateOf = r => r ? stale.has(r.id) ? 'stale' : r.state : 'unreported';
  const date = value => new Date(value).toLocaleString(language === 'zh' ? 'zh-CN' : 'en-GB', {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
  function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5000); }
  function error(message) { $('load-error').textContent = message; $('load-error').hidden = !message; }
  function download(value, filename) {
    const url = URL.createObjectURL(new Blob([typeof value === 'string' ? value : JSON.stringify(value,null,2) + '\n'], {type:'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = filename; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function combined(base, local) {
    const merged = HOI.mergeRecords(base.records, local);
    const errors = [...merged.errors, ...HOI.validateBoard({...base,records:merged.records})];
    if (errors.length) throw new Error(errors.slice(0,8).join('\n'));
    return merged.records;
  }
  function persist(next) {
    records = combined(board,next); drafts = next; stale = HOI.staleRecordIds(records);
    try { localStorage.setItem(draftKey,JSON.stringify({schema_version:'hoi_progress.v1',records:drafts})); }
    catch { toast(t('storageError')); return false; }
    return true;
  }
  function parseImport(text) {
    const parsed = HOI.parseImport(text);
    if (parsed.errors.length) throw new Error(parsed.errors.join('\n'));
    return parsed.records;
  }
  function setBoard(next, isRemote) {
    const problems = HOI.validateBoard(next);
    if (problems.length) throw new Error(problems.slice(0,5).join('\n'));
    const merged = combined(next,drafts);
    board = next; records = merged; remote = isRemote; stale = HOI.staleRecordIds(records);
    // Records already committed are no longer local drafts; collisions were checked above.
    const committed = new Set(board.records.map(r => r.id));
    drafts = drafts.filter(r => !committed.has(r.id));
    if (!recovery) { try { localStorage.setItem(draftKey, JSON.stringify({schema_version:'hoi_progress.v1',records:drafts})); } catch {} }
    render();
  }
  async function fetchJson(url) {
    const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(),15000);
    try { const response = await fetch(url,{cache:'no-store',signal:controller.signal}); if (!response.ok) throw new Error(`HTTP ${response.status}`); return await response.json(); }
    finally { clearTimeout(timeout); }
  }
  function options(id, items) {
    const old = $(id).value; $(id).innerHTML = items.map(([value,label]) => `<option value="${esc(value)}">${esc(label)}</option>`).join('');
    if (items.some(([v]) => v === old)) $(id).value = old;
  }
  function populateFilters() {
    const seen = new Set(); const objects = board.episodes.filter(ep => !seen.has(ep.object) && seen.add(ep.object));
    options('object-filter', [['',t('allObjects')],...objects.map(ep => [ep.object,objectName(ep)])]);
    options('scope-filter', [['',t('allVideos')],['baseline',t('baseline')],['devmini',t('devmini')]]);
    options('stage-filter', [['',t('allStages')],...board.stages.map(s => [s.id,stageName(s.id)])]);
    options('state-filter', [['',t('allStates')],...['unreported',...HOI.STATES,'stale'].map(s => [s,t(s)])]);
    options('iteration-filter', [['',t('allIterations')],...[...new Set(records.map(r => r.iteration))].sort().map(x => [x,x])]);
  }
  function stageCell(ep,stage) {
    const r = latest.get(`${ep.id}:${stage.id}`), status = stateOf(r);
    return `<button class="stage-cell ${status}" data-episode="${ep.id}" data-stage="${stage.id}" aria-label="${esc(`EP ${String(ep.ep).padStart(2,'0')} · ${stageName(stage.id)} · ${t(status)}`)}" title="${esc(r ? `${r.iteration} · ${r.summary}` : t('noRecord'))}"><i class="dot"></i>${esc(t(status))}${r && draftIds().has(r.id) ? '<i class="draft-marker" title="'+esc(t('local'))+'"></i>' : ''}${r && r.kind !== 'real_track1' ? `<span class="scope-mini" title="${esc(t(r.kind))}">◇</span>` : ''}</button>`;
  }
  function identity(ep) {
    return `<div class="video-identity"><img class="thumb" src="${esc(ep.poster_url)}" loading="lazy" alt="${esc(objectName(ep))}"><div class="video-name"><strong>EP ${String(ep.ep).padStart(2,'0')} <span>${esc(objectName(ep))}</span></strong><small title="${esc(ep.task)}">${esc(ep.task)}</small></div></div>`;
  }
  function renderMatrix() {
    latest = HOI.latestRecords(records,iteration());
    const selected = $('stage-filter').value;
    const stages = board.stages.filter(s => !selected || s.id === selected);
    const search = $('search').value.trim().toLowerCase(), object = $('object-filter').value, scope = $('scope-filter').value, status = $('state-filter').value;
    const needs = r => ['blocked','failed','submitted','stale'].includes(stateOf(r));
    const episodes = board.episodes.filter(ep => {
      if (object && ep.object !== object) return false;
      if (scope && !(scope === 'baseline' ? [12,16] : [0,6,9,16,24]).includes(ep.ep)) return false;
      if (search && !`${ep.id} ep ${String(ep.ep).padStart(2,'0')} ${ep.ep} ${ep.object_name} ${objectName(ep)} ${ep.task}`.toLowerCase().includes(search)) return false;
      return stages.some(s => { const r = latest.get(`${ep.id}:${s.id}`); return (!status || stateOf(r) === status) && (!attention || needs(r)); });
    });
    $('result-count').textContent = `${episodes.length} / 30 ${t('resultUnit')}`;
    $('no-results').hidden = episodes.length > 0;
    $('matrix-container').hidden = view !== 'matrix' || !episodes.length;
    $('cards-container').hidden = view !== 'cards' || !episodes.length;
    $('table-view').classList.toggle('selected',view === 'matrix'); $('card-view').classList.toggle('selected',view === 'cards');
    $('nav-board').classList.toggle('active',!attention); $('nav-attention').classList.toggle('active',attention);
    $('nav-board').setAttribute('aria-pressed',String(!attention)); $('nav-attention').setAttribute('aria-pressed',String(attention));
    document.querySelector('#matrix thead').innerHTML = `<tr><th>${esc(t('videos'))}</th><th>${esc(t('frames'))}</th>${stages.map(s => `<th><span>${esc(stageName(s.id))}</span><small>${esc(t(s.module+'Module').split(' · ')[0])}</small></th>`).join('')}<th>${esc(t('acceptedShort'))}</th></tr>`;
    const acceptedCount = ep => board.stages.filter(s => {const r=latest.get(`${ep.id}:${s.id}`);return r?.kind==='real_track1' && stateOf(r)==='accepted';}).length;
    document.querySelector('#matrix tbody').innerHTML = episodes.map(ep => `<tr><td>${identity(ep)}</td><td class="frame-count">${ep.frames}<small>${ep.seconds.toFixed(1)} ${esc(t('seconds'))}</small></td>${stages.map(s => `<td>${stageCell(ep,s)}</td>`).join('')}<td><div class="row-delivery">${acceptedCount(ep)}<span>/ 6</span></div></td></tr>`).join('');
    $('cards-container').innerHTML = episodes.map(ep => `<article class="video-card">${identity(ep)}<div class="card-stages">${stages.map(s => `<div><small>${esc(stageName(s.id))}</small>${stageCell(ep,s)}</div>`).join('')}</div><div class="card-foot"><span>${ep.frames} ${esc(t('frames'))} · ${ep.seconds.toFixed(1)} ${esc(t('seconds'))}</span><span>${acceptedCount(ep)}/6 ${esc(t('acceptedShort'))}</span></div></article>`).join('');
    const real = [...latest.values()].filter(r => r.kind === 'real_track1');
    $('stat-registered').textContent = real.length; $('registered-bar').style.width = `${real.length/180*100}%`;
    $('stat-review').textContent = real.filter(r => stateOf(r) === 'submitted').length;
    $('stat-accepted').textContent = HOI.acceptedEpisodeIds(records,iteration()).size;
    $('attention-count').textContent = [...latest.values()].filter(needs).length;
  }
  function render() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach(el => el.textContent = t(el.dataset.i18n));
    document.querySelectorAll('[data-placeholder]').forEach(el => el.placeholder = t(el.dataset.placeholder));
    $('language').textContent = language === 'zh' ? 'EN' : '中文';
    if (!board) return;
    $('source-label').textContent = t(remote ? 'sourceRemote' : 'sourceSnapshot');
    $('draft-count').textContent = drafts.length;
    $('context-message').textContent = t(records.length ? 'contextActive':'contextEmpty') + (drafts.length ? t('contextDraft') : '') + (!iteration() && records.length ? ' '+t('allViewHint') : '');
    $('snapshot-info').textContent = `${t('snapshot')}: ${date(board.updated_at)} · ${t('basis')}: ${board.source.revision.slice(0,7)}`;
    $('module-links').innerHTML = ['platform','human','object','physics'].map(m => `<a class="module-link" href="${repo}/tree/main/deliveries/${m}" target="_blank" rel="noopener"><span class="module-bullet ${m}"></span>${esc(t(m+'Module'))}<span>↗</span></a>`).join('');
    $('legend').innerHTML = ['unreported','running','submitted','checked','accepted','stale','blocked'].map(s => `<span class="${s}"><i class="dot"></i>${esc(t(s))}</span>`).join('') + `<span><i class="draft-marker"></i>${esc(t('local'))}</span>`;
    $('capability-grid').innerHTML = board.capabilities.map(c => `<article class="capability-card"><div class="capability-heading"><i class="module-bullet ${c.module}"></i>${esc(t(c.module+'Module'))}<b>FOUNDATION</b></div><p>${esc(language === 'zh' ? capabilityZh[c.module] : c.summary)}</p>${link(c.evidence_url,t('evidence'))}</article>`).join('');
    populateFilters(); renderMatrix();
    if ($('detail').open && !editing) renderDetail();
    if ($('guide').open) renderGuide();
  }
  function openDetail(episodeId, stageId) {
    selectedEpisode = board.episodes.find(e => e.id === episodeId); selectedStage = board.stages.find(s => s.id === stageId); selectedRecordId = null; editing = false;
    renderDetail(); if (!$('detail').open) $('detail').showModal(); $('detail').scrollTop = 0;
  }
  function renderDetail() {
    const ep = selectedEpisode, stage = selectedStage;
    const r = selectedRecordId ? records.find(r => r.id === selectedRecordId) : latest.get(`${ep.id}:${stage.id}`);
    $('detail-heading').innerHTML = `<div class="eyebrow">${esc(ep.id)} · ${esc(t(stage.module+'Module'))}</div><h2 id="detail-title">${esc(objectName(ep))} / ${esc(stageName(stage.id))}</h2>`;
    const history = records.filter(x => x.episode_id === ep.id && x.stage === stage.id).sort((a,b) => Date.parse(b.created_at)-Date.parse(a.created_at) || (a.id < b.id ? 1 : a.id > b.id ? -1 : 0));
    const artifacts = new Map((r?.artifacts || []).map(a => [a.requirement_id,a]));
    const missing = HOI.missingRequirements(r,stage);
    $('detail-content').innerHTML = `<div class="detail-video"><img src="${esc(ep.poster_url)}" alt="${esc(objectName(ep))}"><div><p>${ep.frames} ${esc(t('frames'))} · ${ep.seconds.toFixed(1)} ${esc(t('seconds'))}<br>${esc(ep.camera)}</p>${link(ep.video_url,t('sourceVideo'))}</div></div><details class="task-description"><summary>${esc(t('metadataTask'))}</summary><p>${esc(ep.task)}</p></details><div class="stage-tabs">${board.stages.map(s => `<button data-tab="${s.id}" class="${s.id===stage.id?'selected':''}">${esc(stageName(s.id))}</button>`).join('')}</div>
      <div class="detail-status"><span class="stage-cell ${stateOf(r)}"><i class="dot"></i>${esc(t(stateOf(r)))}</span><span class="scope-chip">${esc(r ? `${r.iteration} · ${t(r.kind)}` : t('unreported'))}</span>${r && draftIds().has(r.id) ? `<span class="local-label">${esc(t('local'))}</span>`:''}</div>
      ${selectedRecordId && selectedRecordId !== latest.get(`${ep.id}:${stage.id}`)?.id ? `<p class="muted">${esc(t('historical'))} · ${esc(r.id)} <button class="text-button" id="view-latest">${esc(t('viewLatest'))}</button></p>`:''}
      <p class="detail-summary">${esc(r?.summary || t('noRecord'))}</p>${r && stale.has(r.id) ? `<div class="notice error">${esc(t('staleNotice'))}</div>` : ''}
      <div class="detail-label"><h3>${esc(t('requirements'))}</h3><small>${stage.requirements.length-missing.length} / ${stage.requirements.length} ${esc(t('present'))}</small></div>
      <div class="artifact-list">${stage.requirements.map(req => {const a=artifacts.get(req.id);return `<div class="artifact-row ${a?'present':''}"><span class="artifact-icon">${a?'✓':'○'}</span><div><strong>${esc(reqName(req))}</strong>${a ? `<small>${esc(a.label)} · ${Number(a.bytes).toLocaleString()} bytes</small><small>SHA-256: ${esc(a.sha256)}</small>${link(a.url,t('artifact'))}` : `<small>${esc(req.id)}</small>`}</div>${!a?`<span class="missing">${esc(t('missing'))}</span>`:''}</div>`;}).join('')}</div>
      <div class="detail-label"><h3>${esc(t('evidence'))}</h3></div><div class="check-grid">${['structural','quality','export'].map(k => {const c=r?.checks[k];return `<div class="check-card ${c?.status||''}">${esc(t(k==='export'?'exportCheck':k))}<strong>${esc(t(c?.status||'not_run'))}</strong>${c?.evidence_url?link(c.evidence_url,t('evidence')):''}</div>`;}).join('')}</div><p class="small-note">${esc(t('checksNote'))}</p>
      <div class="detail-key-values">${[[t('contributor'),r?.contributor],[t('reviewer'),r?.reviewer],[t('coverage'),`${r?.coverage.frames??'?'} / ${ep.frames}`],[t('objectCoverage'),t(r?.coverage.objects_complete===true?'yes':r?.coverage.objects_complete===false?'no':'unknown')],[t('commit'),r?.source_commit],[t('currentRecord'),r?.id]].map(([key,value]) => `<div><span>${esc(key)}</span><b>${esc(value||'—')}</b></div>`).join('')}</div>
      <div class="detail-label"><h3>${esc(t('upstream'))}</h3></div><div class="upstream-list">${stage.dependencies.length ? stage.dependencies.map(id => `<div><span>${esc(stageName(id))}</span><code>${esc(r?.upstream.find(p => p.stage===id)?.record_id || t('unbound'))}</code></div>`).join('') : `<p class="muted">${esc(t('noUpstream'))}</p>`}</div>
      ${r?.blocker?`<div class="detail-label"><h3>${esc(t('blocker'))}</h3></div><p class="detail-summary">${esc(r.blocker)}</p>`:''}${r?.next_action?`<div class="detail-label"><h3>${esc(t('nextAction'))}</h3></div><p class="detail-summary">${esc(r.next_action)}</p>`:''}
      <div class="actions-row"><button id="new-record" class="button primary">${esc(t('newRecord'))}</button>${r && draftIds().has(r.id)?`<button id="remove-draft" data-record="${esc(r.id)}" class="button">${esc(t('removeDraft'))}</button>`:''}</div><div class="detail-label">${link(`${repo}/tree/main/deliveries/${stage.module}`,t('moduleLocation'))}${r?.delivery_url ? link(r.delivery_url,t('deliveryRecord')):''}</div>
      <div class="detail-label"><h3>${esc(t('history'))}</h3><small>${history.length}</small></div>${history.map(x => `<div class="history-entry"><div><b>${esc(x.iteration)} · ${esc(t(stateOf(x)))}</b> ${draftIds().has(x.id)?`<span class="local-label">${esc(t('local'))}</span>`:''}<p>${esc(x.summary)}</p><p>${esc(x.contributor)} · ${esc(date(x.created_at))}</p><code>${esc(x.id)}</code></div><button class="text-button" data-history="${esc(x.id)}">${esc(t('viewRecord'))}</button></div>`).join('') || `<p class="muted">${esc(t('historyEmpty'))}</p>`}`;
  }
  function input(label,name,value='',type='text',wide=false,extra='') {return `<label class="field ${wide?'wide':''}"><span>${esc(label)}</span><input type="${type}" name="${esc(name)}" value="${esc(value)}" ${extra}></label>`;}
  function select(label,name,items,value,wide=false) {return `<label class="field ${wide?'wide':''}"><span>${esc(label)}</span><select name="${esc(name)}">${items.map(([v,l])=>`<option value="${esc(v)}" ${v===value?'selected':''}>${esc(l)}</option>`).join('')}</select></label>`;}
  function textarea(label,name,value='') {return `<label class="field wide"><span>${esc(label)}</span><textarea name="${esc(name)}">${esc(value)}</textarea></label>`;}
  function renderForm() {
    editing = true; const ep=selectedEpisode, stage=selectedStage;
    const r=selectedRecordId ? records.find(x=>x.id===selectedRecordId) : latest.get(`${ep.id}:${stage.id}`);
    const id = `ep${String(ep.ep).padStart(2,'0')}-${stage.id}-${Date.now()}-${crypto.getRandomValues(new Uint32Array(1))[0].toString(36)}`;
    $('detail-content').innerHTML = `<p class="detail-subtitle">${esc(t('draftHint'))}</p><form id="record-form"><div class="form-grid">
      ${input(t('recordId'),'id',id,'text',true,'required maxlength="96"')}${input(t('iteration'),'iteration',iteration()||r?.iteration||'baseline-v1','text',false,'required')}${input(t('contributor'),'contributor',r?.contributor||'','text',false,'required')}
      ${select(t('state'),'state',HOI.STATES.map(s=>[s,t(s)]),'planned')}${select(t('kind'),'kind',HOI.KINDS.map(k=>[k,t(k)]),r?.kind||'real_track1')}${textarea(t('summary'),'summary')}${textarea(t('nextAction'),'next_action')}${textarea(t('blocker'),'blocker')}
      ${input(t('deliveryUrl'),'delivery_url','','url',true)}${input(t('sourceCommit'),'source_commit','','text',true)}${input(t('reviewer'),'reviewer')}${input(t('framesCovered'),'frames','','number',false,'min="0" step="1"')}${select(t('objectsComplete'),'objects_complete',[['',t('unknown')],['true',t('yes')],['false',t('no')]],'')}
      </div><details class="form-section"><summary>${esc(t('requirements'))} (${stage.requirements.length})</summary><p class="hint">${esc(t('artifactHint'))}</p>${stage.requirements.map(req=>`<div class="artifact-fields"><h4>${esc(reqName(req))}</h4><div class="form-grid">${input(t('url'),`artifact_${req.id}_url`,'','url',true)}${input(t('hash'),`artifact_${req.id}_sha256`,'','text',true)}${input(t('bytes'),`artifact_${req.id}_bytes`,'','number',true,'min="1" step="1"')}</div></div>`).join('')}</details>
      <details class="form-section"><summary>${esc(t('evidence'))}</summary><p class="hint">${esc(t('checksNote'))}</p>${['structural','quality','export'].map(k=>`<div class="check-edit">${select(t(k==='export'?'exportCheck':k),`check_${k}_status`,['not_run','pass','fail'].map(s=>[s,t(s)]),'not_run')}${input(t('checkUrl'),`check_${k}_url`,'','url')}</div>`).join('')}</details>
      <details class="form-section"><summary>${esc(t('upstream'))}</summary><p class="hint">${esc(t('upstreamHint'))}</p><div class="form-grid">${stage.dependencies.map(dep=>select(stageName(dep),`upstream_${dep}`,[['',t('unbound')],...records.filter(x=>x.episode_id===ep.id&&x.stage===dep).sort((a,b)=>Date.parse(b.created_at)-Date.parse(a.created_at)).map(x=>[x.id,`${x.iteration} · ${t(stateOf(x))} · ${x.id}`])],'',true)).join('')||`<p class="muted">${esc(t('noUpstream'))}</p>`}</div></details>
      <div id="form-errors" class="form-error" role="alert" hidden></div><div class="actions-row"><button type="submit" class="button primary">${esc(t('saveDraft'))}</button><button id="cancel-edit" type="button" class="button">${esc(t('cancel'))}</button></div></form>`;
    $('record-form').addEventListener('submit',saveForm); $('detail').scrollTop=0; $('record-form').elements.namedItem('contributor').focus();
  }
  function saveForm(event) {
    event.preventDefault(); if(recovery) { $('form-errors').hidden=false; $('form-errors').textContent=t('invalidStorage'); return; }
    const form = new FormData(event.currentTarget), get = name => String(form.get(name)||'').trim();
    const record = {id:get('id'),episode_id:selectedEpisode.id,stage:selectedStage.id,iteration:get('iteration'),state:get('state'),kind:get('kind'),created_at:new Date().toISOString(),contributor:get('contributor'),reviewer:get('reviewer'),summary:get('summary'),blocker:get('blocker'),next_action:get('next_action'),delivery_url:get('delivery_url'),source_commit:get('source_commit'),coverage:{frames:get('frames')===''?null:Number(get('frames')),objects_complete:get('objects_complete')===''?null:get('objects_complete')==='true'},upstream:selectedStage.dependencies.filter(s=>get(`upstream_${s}`)).map(s=>({stage:s,record_id:get(`upstream_${s}`)})),artifacts:selectedStage.requirements.filter(req=>['url','sha256','bytes'].some(k=>get(`artifact_${req.id}_${k}`))).map(req=>({requirement_id:req.id,label:req.label,url:get(`artifact_${req.id}_url`),sha256:get(`artifact_${req.id}_sha256`),bytes:Number(get(`artifact_${req.id}_bytes`))})),checks:Object.fromEntries(['structural','quality','export'].map(k=>[k,{status:get(`check_${k}_status`),evidence_url:get(`check_${k}_url`)}]))};
    try { const next = HOI.mergeRecords(drafts,[record]); if(next.errors.length)throw new Error(next.errors.join('\n')); const saved=persist(next.records); editing=false; selectedRecordId=record.id; render(); renderDetail(); toast(t(saved?'saved':'storageError')); }
    catch(e){ $('form-errors').textContent=t('validationError')+'\n'+e.message; $('form-errors').hidden=false; $('form-errors').scrollIntoView({block:'nearest'}); }
  }
  function renderGuide() {
    $('guide-content').innerHTML = `<p class="detail-subtitle">${esc(t('guideIntro'))}</p>${[1,2,3].map(i=>`<div class="guide-step"><b>${i}</b><div><h3>${esc(t('guide'+i))}</h3><p>${esc(t('guide'+i+'Body'))}</p></div></div>`).join('')}<pre class="command">python tools/progress_board.py import /path/to/hoi-progress-drafts.json
python tools/progress_board.py validate
git add tracking/records tracking/board.json
git commit -m "Register video delivery progress"
git push origin HEAD</pre><div class="guide-step"><b>4</b><div><h3>${esc(t('guide4'))}</h3><p>${esc(t('guide4Body'))}</p></div></div><p class="small-note">${esc(t('privateNote'))}</p><div class="detail-label">${link(`${repo}/blob/main/tracking/README.md`,t('deliveryGuide').replace(' ↗',''))}</div><p class="small-note">${esc(t('localPreview'))}</p><pre class="command">python -m http.server 8765
http://localhost:8765/docs/progress-board/</pre>`;
  }
  function resetFilters(){['object-filter','scope-filter','stage-filter','state-filter','iteration-filter'].forEach(id=>$(id).value='');$('search').value='';attention=false;render();}
  $('language').addEventListener('click',()=>{if(editing){toast(language==='zh'?'请先保存或取消当前表单。':'Save or cancel the current form first.');return;} language=language==='zh'?'en':'zh';render();});
  ['object-filter','scope-filter','stage-filter','state-filter','iteration-filter'].forEach(id=>$(id).addEventListener('change',()=>board&&render()));
  $('search').addEventListener('input',()=>board&&renderMatrix()); $('clear-filters').addEventListener('click',resetFilters);
  $('table-view').addEventListener('click',()=>{view='matrix';if(board)renderMatrix();}); $('card-view').addEventListener('click',()=>{view='cards';if(board)renderMatrix();});
  $('board-panel').addEventListener('click',event=>{const cell=event.target.closest('[data-episode][data-stage]');if(cell)openDetail(cell.dataset.episode,cell.dataset.stage);});
  $('nav-board').addEventListener('click',()=>{attention=false;if(board)render();$('board-panel').scrollIntoView({block:'start'});});
  $('nav-attention').addEventListener('click',()=>{attention=true;if(board)render();$('board-panel').scrollIntoView({block:'start'});});
  $('nav-capabilities').addEventListener('click',()=>$('capabilities').scrollIntoView({behavior:'smooth'}));
  $('close-detail').addEventListener('click',()=>$('detail').close()); $('close-guide').addEventListener('click',()=>$('guide').close());
  $('detail').addEventListener('close',()=>{editing=false;});
  $('guide-button').addEventListener('click',()=>{renderGuide();$('guide').showModal();});
  $('detail-content').addEventListener('click',event=>{
    const button=event.target.closest('button');if(!button)return;
    if(button.dataset.tab)openDetail(selectedEpisode.id,button.dataset.tab);
    else if(button.dataset.history){selectedRecordId=button.dataset.history;renderDetail();$('detail').scrollTop=0;}
    else if(button.id==='new-record')renderForm();
    else if(button.id==='cancel-edit'){editing=false;renderDetail();}
    else if(button.id==='view-latest'){selectedRecordId=null;renderDetail();}
    else if(button.id==='remove-draft') {try{const saved=persist(drafts.filter(r=>r.id!==button.dataset.record));selectedRecordId=null;render();renderDetail();toast(t(saved?'removed':'storageError'));}catch(e){toast(e.message);}}
  });
  $('export').addEventListener('click',()=>{if(recovery){download(recovery,'hoi-progress-recovery.json');return;}if(!drafts.length){toast(t('noDrafts'));return;}download({schema_version:'hoi_progress.v1',records:drafts},'hoi-progress-drafts.json');toast(t('exported'));});
  $('import-button').addEventListener('click',()=>$('import-file').click());
  $('import-file').addEventListener('change',async event=>{
    const file=event.target.files[0];event.target.value='';if(!file)return;
    try {if(!board)throw new Error(t('loadError'));if(recovery)throw new Error(t('invalidStorage'));if(file.size>5*1024*1024)throw new Error('JSON limit: 5 MiB');const incoming=parseImport(await file.text());const merged=HOI.mergeRecords(drafts,incoming);if(merged.errors.length)throw new Error(merged.errors.join('\n'));combined(board,merged.records);const canonical=new Set(board.records.map(r=>r.id));const saved=persist(merged.records.filter(r=>!canonical.has(r.id)));render();toast(t(saved?'imported':'storageError'));}catch(e){toast(t('importError')+e.message);}
  });
  $('refresh').addEventListener('click',async()=>{
    $('refresh').disabled=true;$('refresh').textContent=t('refreshing');
    try{setBoard(await fetchJson(raw),true);error(recovery?t('invalidStorage'):'');toast(t('refreshed'));}catch(e){error(`${t('loadError')} ${e.message}`);}finally{$('refresh').disabled=false;$('refresh').textContent=t('refresh');}
  });
  async function start() {
    render();
    let snapshot; const paths = location.pathname.includes('/docs/progress-board/') ? ['../../tracking/board.json','./board.json'] : ['./board.json'];
    for(const path of paths) {try{const candidate=await fetchJson(path);const errors=HOI.validateBoard(candidate);if(errors.length)throw new Error(errors[0]);snapshot=candidate;break;}catch{}}
    if(!snapshot){error(t('loadError'));return;}
    try { const stored=localStorage.getItem(draftKey);if(stored){try{const local=parseImport(stored);combined(snapshot,local);drafts=local;}catch{recovery=stored;error(t('invalidStorage'));}}}catch{error(t('storageError'));}
    setBoard(snapshot,false);
  }
  start().catch(e=>error(`${t('loadError')} ${e.message}`));
})();
