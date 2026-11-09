import { el, esc, api, showError } from './core.js?v=95';
import { hideHoverPreview } from './preview.js?v=95';
import { loadImageGallery, gallerySelection } from './images.js?v=111';
import { followBatch } from './aggregate-batch.js?v=111';
import { setupAggregateHistory, loadAggregateHistory } from './aggregate-history.js?v=114';
import { classColor } from './viewer.js?v=95';
import { renderAggregateProgress, followAggregateJob } from './aggregate-progress.js?v=111';

/* ==================== 聚合推理（目标检测 → 人脸识别 → 姿势检测绑定）==================== */

const STEP_LABELS = { detect: '目标检测', face: '人脸识别', pose: '姿势检测', multimodal: 'VLM 动作分析' };
/** 只画 head 框；筛选按「已贴姓名 / 未识别 / 已绑姿势」分档，与图片推理的类别胶囊同一套交互 */
const FILTERS = [
  { key: 'all', label: '全部 head' },
  { key: 'named', label: '已贴姓名' },
  { key: 'unknown', label: '未识别' },
  { key: 'posed', label: '已有姿势 / 动作' },
];
const AGG_PANES = { images: 'aggregate-pane-images', history: 'aggregate-pane-history', result: 'aggregate-pane-result' };
let ACTIVE_RECORD = null;
let ACTIVE_FILTER = 'all';
let ACTIVE_STATE = 'all';
let ALWAYS_TAGS = false;
let SHOW_POINTS = false;
let AGG_BUSY = false;
let ACTIVE_JOB = null;
let STOP_PENDING = false;

function syncAggregateControls() {
  const selection = gallerySelection('aggregate');
  const busy = AGG_BUSY || selection.busy;
  el('aggBatchStart').disabled = busy || !selection.selected.length;
  el('aggStop').hidden = !AGG_BUSY;
  el('aggStop').disabled = !ACTIVE_JOB || STOP_PENDING;
  el('aggStop').textContent = STOP_PENDING ? '正在停止…' : '停止推理';
  el('aggImageDeleteSelected').disabled = busy || !selection.selected.length;
  el('aggImageClearSelection').disabled = busy || !selection.selected.length;
  el('aggImageSelectAll').disabled = busy || !selection.count;
  el('aggImageGrid').inert = AGG_BUSY;
  el('aggFaceRecognition').disabled = AGG_BUSY;
  el('aggMultimodal').disabled = AGG_BUSY;
  el('aggImageGrid').querySelectorAll('input, button').forEach(node => { node.disabled = busy; });
}

function setAggregateBusy(value) {
  AGG_BUSY = value;
  syncAggregateControls();
}

function syncMultimodal() {
  const multimodal = el('aggMultimodal');
  el('aggMultimodalHint').hidden = !multimodal.checked;
}

function hasAction(target) {
  return target.action?.status === 'done' || (target.poses || []).length > 0;
}

function targetStates(target) {
  if (target.action) return target.action.state ? [target.action.state] : [];
  return [...new Set((target.poses || []).map(pose => pose.behavior?.state).filter(Boolean))];
}

function showAggregateError(error, filename) {
  if (error.job) renderAggregateProgress(error.job, filename);
  else showError(el('aggResult'), error, '聚合失败');
}

/* ==================== 子标签页（待推理图片 / 聚合结果，布局与图片推理一致）==================== */

/** pane: 'images' | 'result' */
function switchAggregatePane(pane) {
  Object.entries(AGG_PANES).forEach(([key, id]) => el(id).classList.toggle('active', key === pane));
  document.querySelectorAll('.subtab[data-agg]').forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.agg === pane);
  });
  el('aggRefresh').hidden = pane === 'result';
  if (pane === 'history') loadAggregateHistory();
  if (pane !== 'images') hideHoverPreview();
  window.scrollTo(0, 0);
}

document.querySelectorAll('.subtab[data-agg]').forEach((tab) => {
  tab.addEventListener('click', () => switchAggregatePane(tab.dataset.agg || 'images'));
});

/** 三步各自的耗时与统计（未勾选的显示「未启用」，避免看起来像漏跑） */
function stepLine(steps) {
  return Object.entries(STEP_LABELS).map(([key, label]) => {
    const step = (steps || {})[key];
    if (key === 'pose' && !step?.enabled) return '';
    if (!step || !step.enabled) return `<span class="agg-step is-off">${esc(label)}：${esc(step?.reason || '未启用')}</span>`;
    const detail = key === 'detect' ? `${step.count} 个人头`
      : key === 'face' ? `检出 ${step.face_count} 张 · 识别 ${step.matched} · 库内 ${step.library_count} 人 · 阈值 ${step.threshold}`
        : key === 'multimodal' ? `完成 ${step.completed}/${step.count} 个目标 · 失败 ${step.failed} · 提示词 ${step.prompt_config_id.slice(0, 8)}`
        : `${step.count} 人 · 绑定 ${step.bound} · 未绑定 ${step.unbound}`;
    return `<span class="agg-step"><b>${esc(label)}</b>：${esc(detail)}｜${esc(step.elapsed_seconds)}s</span>`;
  }).join('');
}

/** head 框上的标签：姓名 + 相似度（未识别就写类别 + 置信度），再挂上姿势状态 */
function headLabel(target) {
  const states = target.action ? [target.action.state || '动作分析失败']
    : targetStates(target);
  const poseNote = states.length ? ` · ${states.join('、')}` : '';
  return target.name
    ? `${target.name} ${target.similarity}${poseNote}`
    : `${target.class_name} ${target.confidence}${poseNote}`;
}

function visibleTargets() {
  let targets = ACTIVE_RECORD?.targets || [];
  if (ACTIVE_FILTER === 'named') targets = targets.filter(target => target.name);
  if (ACTIVE_FILTER === 'unknown') targets = targets.filter(target => !target.name);
  if (ACTIVE_FILTER === 'posed') targets = targets.filter(hasAction);
  return ACTIVE_STATE === 'all' ? targets : targets.filter(target => targetStates(target).includes(ACTIVE_STATE.slice(6)));
}

/** 框用百分比定位（与 viewer.js 一致：viewer 尺寸 = 图片显示尺寸） */
function pct(value, total) {
  return `${(value / Math.max(1, total)) * 100}%`;
}

/** 只画 head 框，姓名与姿势状态都在框的标签上；明细表里同一行跟着显隐 */
function paintHeads() {
  const record = ACTIVE_RECORD;
  const box = el('aggResult');
  if (!record) return;
  const layer = box.querySelector('[data-role="boxes"]');
  if (!layer) return;
  box.querySelector('[data-role="viewer"]').classList.toggle('always-tags', ALWAYS_TAGS);
  layer.innerHTML = '';
  const shown = visibleTargets();
  shown.forEach((target) => {
    const [x1, y1, x2, y2] = target.bbox;
    const node = document.createElement('div');
    node.className = 'bbox';
    node.style.setProperty('--c', target.name ? '#137333' : '#1a73e8');
    node.style.left = pct(x1, record.image_width);
    node.style.top = pct(y1, record.image_height);
    node.style.width = pct(x2 - x1, record.image_width);
    node.style.height = pct(y2 - y1, record.image_height);
    if (SHOW_POINTS) {
      node.classList.add('aggregate-point');
      node.style.left = pct((x1 + x2) / 2, record.image_width);
      node.style.top = pct((y1 + y2) / 2, record.image_height);
      node.style.width = '9px';
      node.style.height = '9px';
    }
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = headLabel(target);
    node.appendChild(tag);
    // hover 已识别的 head 框 → 在框旁边浮出人脸库登记照（对应名字），便于人工核对
    if (target.name && target.photo_url) {
      let libImg = null;
      node.addEventListener('mouseenter', () => {
        if (libImg) return;
        libImg = document.createElement('img');
        libImg.className = 'agg-lib-photo';
        libImg.src = target.photo_url;
        libImg.alt = `${target.name} 登记照`;
        libImg.title = `${target.name}（编号 ${target.entry_id || ''}）人像库登记照`;
        node.appendChild(libImg);
      });
      node.addEventListener('mouseleave', () => {
        if (!libImg) return;
        if (libImg.remove) libImg.remove();
        else if (libImg.parentNode) libImg.parentNode.removeChild(libImg);
        libImg = null;
      });
    }
    layer.appendChild(node);
  });
  box.querySelectorAll('[data-role="headrow"]').forEach((row) => {
    row.hidden = !shown.includes(record.targets[Number(row.dataset.i)]);
  });
  const counts = {
    all: record.targets.length,
    named: record.targets.filter((target) => target.name).length,
    unknown: record.targets.filter((target) => !target.name).length,
    posed: record.targets.filter(hasAction).length,
  };
  box.querySelectorAll('[data-role="aggfilter"] .chip').forEach((chip) => {
    chip.textContent = `${chip.dataset.label} ${counts[chip.dataset.filter]}`;
    chip.style.opacity = chip.dataset.filter === ACTIVE_FILTER ? '1' : '.5';
  });
  const countNode = box.querySelector('#aggResultCount');
  if (countNode) {
    countNode.textContent =
      `图上 ${shown.length} 个 head 框 / 共检出 ${record.targets.length} 个｜已贴姓名 ${counts.named} 个` +
      (record.steps?.multimodal?.enabled ? `｜VLM 动作 ${record.summary.vlm_actions} 个`
        : record.summary ? `｜姿势绑定 ${record.summary.bound_poses}/${record.summary.poses}` : '');
  }
}

function renderAggregateResult(record, filename) {
  const box = el('aggResult');
  ACTIVE_RECORD = record;
  ACTIVE_FILTER = 'all';
  ACTIVE_STATE = 'all';
  const summary = record.summary || {};
  const stateCounts = new Map();
  for (const target of record.targets || []) {
    for (const state of targetStates(target)) stateCounts.set(state, (stateCounts.get(state) || 0) + 1);
  }
  el('aggResultName').textContent = filename;
  el('aggStatus').textContent =
    `人头 ${record.count} 个｜贴姓名 ${summary.named_targets ?? 0} 个｜` +
    `人脸 ${summary.faces ?? 0} 张（已识别 ${summary.matched_faces ?? 0}、未识别 ${summary.unknown_faces ?? 0}、质量不足 ${summary.rejected_faces ?? 0}）｜` +
    (record.steps?.multimodal?.enabled ? `VLM 动作 ${summary.vlm_actions ?? 0} 个（失败 ${summary.failed_actions ?? 0}）`
      : `姿势 ${summary.poses ?? 0} 人（已绑定 ${summary.bound_poses ?? 0}）`) + `｜总耗时 ${record.elapsed_seconds}s`;

  const notes = [];
  if ((record.unbound_faces || []).length) notes.push(`没落到 head 框的人脸：${record.unbound_faces.join('、')}`);
  if ((record.unbound_poses || []).length) notes.push(`没绑上 head 框的姿势：第 ${record.unbound_poses.join('、')} 人`);
  const rows = (record.targets || []).map((target, index) => {
    const [x1, y1, x2, y2] = target.bbox;
    const poses = target.action ? `<b>${esc(target.action.state || '分析失败')}</b><br><small>VLM · ${esc(target.action.evidence || target.action.error || '')}</small>`
      : (target.poses || []).map((pose) =>
        `${esc(pose.behavior?.state || '无法判断')} <small>IoU ${esc(pose.bind?.iou)}</small>`).join('、');
    return `<tr data-role="headrow" data-i="${index}">` +
      `<td>${esc(target.class_name)} <small>${esc(target.confidence)}</small></td>` +
      `<td>${target.name ? `<b class="agg-name">${esc(target.name)}</b> <small>${esc(target.similarity)}</small>` : '<span class="agg-muted">—</span>'}</td>` +
      `<td>${poses || '<span class="agg-muted">—</span>'}</td>` +
      `<td>${esc(Math.round((x1 + x2) / 2))}, ${esc(Math.round((y1 + y2) / 2))}</td>` +
      `<td>${esc(x1)}, ${esc(y1)}, ${esc(x2)}, ${esc(y2)}</td></tr>`;
  }).join('');

  box.innerHTML =
    '<div class="aggregate-progress"><label>聚合完成 · 100%<progress aria-label="聚合阶段进度" max="100" value="100"></progress></label></div>' +
    `<div class="agg-steps">${stepLine(record.steps)}</div>` +
    (notes.length ? `<p class="hint">${esc(notes.join('；'))}</p>` : '') +
    `<div class="viewer-bar aggregate-detail-controls">` +
    `<label><input type="checkbox" data-role="alwaystag"> 常显标签</label>` +
    `<label><input type="checkbox" data-role="showpoints"> 显示为点</label>` +
    `<div class="chips" data-role="aggfilter">` +
    FILTERS.map((filter, index) => `<button class="chip" data-filter="${filter.key}" data-label="${esc(filter.label)}"` +
      ` style="background:${classColor(index)}">${esc(filter.label)}</button>`).join('') +
    `</div>` +
    `<label>姿态筛选 <select data-role="actionfilter" ${stateCounts.size ? '' : 'disabled'}>` +
    `<option value="all">全部姿态</option>` +
    [...stateCounts].map(([state, count]) => `<option value="state:${esc(state)}">${esc(state)}（${count}）</option>`).join('') +
    '</select></label>' +
    '<div class="aggregate-help aggregate-detail-help">' +
    '<button class="aggregate-help-button" type="button" aria-label="查看本条聚合结果说明" aria-describedby="aggRecordHelp">?</button>' +
    '<div class="aggregate-help-tooltip" id="aggRecordHelp" role="tooltip"><div class="aggregate-help-content">' +
    `<p>目标检测：${esc(record.steps?.detect?.rule || '')}</p>` +
    `<p>姓名归属与姿势绑定：${esc(record.bind?.rule || '')}。结果只画 head 框，姓名与姿势状态都挂在这个框上；` +
    `没绑上的人脸 / 姿势在清单里列出来，不会硬塞给某个人头。` +
    (record.steps?.multimodal?.enabled ? 'VLM 动作由对应人头裁剪直接归属，无需 Pose 框匹配。' : '') + '</p>' +
    '</div></div></div></div>' +
    `<div class="viewer" data-role="viewer">` +
    `<img class="base" data-role="baseimg" src="${esc(record.source_url || `/images/${encodeURIComponent(filename)}`)}" alt="聚合图">` +
    `<div class="boxes" data-role="boxes"></div></div>` +
    `<div class="viewer-foot"><span id="aggResultCount"></span>` +
    (record.result_url ? `<a href="${esc(record.result_url)}" target="_blank">后端渲染图（对照）</a>` : '') +
    `</div>` +
    `<div class="det-table-wrap"><table><thead><tr><th>人头</th><th>姓名</th><th>姿势 / 动作</th>` +
    `<th>中心 (cx, cy)</th><th>左上右下 (x1, y1, x2, y2)</th></tr></thead><tbody>${rows}</tbody></table></div>`;


  const tagToggle = box.querySelector('[data-role="alwaystag"]');
  tagToggle.checked = ALWAYS_TAGS;
  const pointToggle = box.querySelector('[data-role="showpoints"]');
  pointToggle.checked = SHOW_POINTS;
  pointToggle.addEventListener('change', event => {
    SHOW_POINTS = event.target.checked;
    paintHeads();
  });
  tagToggle.addEventListener('change', event => {
    ALWAYS_TAGS = event.target.checked;
    paintHeads();
  });
  box.querySelector('[data-role="actionfilter"]').addEventListener('change', event => {
    ACTIVE_STATE = event.target.value;
    paintHeads();
  });
  box.querySelectorAll('[data-role="aggfilter"] .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      ACTIVE_FILTER = chip.dataset.filter;
      paintHeads();
    });
  });
  const base = box.querySelector('[data-role="baseimg"]');
  if (base) base.addEventListener('load', paintHeads);
  paintHeads();
}

async function loadAggregate() {
  const gallery = await loadImageGallery('aggregate');
  if (!AGG_BUSY) {
    try {
      const { job } = await api('/api/aggregate/jobs/active');
      if (job) {
        ACTIVE_JOB = job.job_id;
        STOP_PENDING = !!job.cancel_requested;
        setAggregateBusy(true);
        el('aggBatchPanel').hidden = !job.batch;
        switchAggregatePane('result');
        el('aggResultName').textContent = job.source;
        const follow = job.batch ? followBatch(job, renderAggregateResult, trackJob) : followAggregateJob(job, job.source).then(result => { if (result) renderAggregateResult(result, job.source); });
        follow
          .catch(error => showAggregateError(error, job.source))
          .finally(() => { finishJob(); });
      }
    } catch (error) { showError(el('aggResult'), error, '查询聚合任务失败'); }
  }
  return gallery;
}

function trackJob(job) {
  ACTIVE_JOB = job.job_id;
  STOP_PENDING = !!job.cancel_requested;
  syncAggregateControls();
}

function finishJob() {
  ACTIVE_JOB = null;
  STOP_PENDING = false;
  setAggregateBusy(false);
}

el('aggStop').addEventListener('click', async () => {
  if (!ACTIVE_JOB || STOP_PENDING) return;
  STOP_PENDING = true;
  syncAggregateControls();
  try {
    await api(`/api/aggregate/jobs/${ACTIVE_JOB}/cancel`, { method: 'POST' });
  } catch (error) {
    STOP_PENDING = false;
    el('aggStatus').textContent = `停止请求失败：${error.message}`;
    syncAggregateControls();
  }
});

el('aggBatchStart').addEventListener('click', async () => {
  const filenames = gallerySelection('aggregate').selected;
  if (AGG_BUSY || !filenames.length) return;
  setAggregateBusy(true);
  ACTIVE_RECORD = null;
  el('aggStatus').textContent = '';
  el('aggBatchPanel').hidden = true;
  switchAggregatePane('result');
  el('aggResult').textContent = '批量任务正在准备，完成的图片可在上方逐张查看。';
  try {
    const job = await api('/api/aggregate/batch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filenames, multimodal: !!el('aggMultimodal').checked }),
    });
    await followBatch(job, renderAggregateResult, trackJob);
  } catch (error) { showAggregateError(error, filenames[0]); }
  finally { finishJob(); }
});
document.addEventListener('gallery-selection-change', event => {
  if (event.detail.kind === 'aggregate') syncAggregateControls();
});

setupAggregateHistory(row => {
  el('aggBatchPanel').hidden = true;
  renderAggregateResult(row.result, row.source);
  switchAggregatePane('result');
});
el('aggRefresh').addEventListener('click', () => {
  if (el('aggregate-pane-history').classList.contains('active')) return loadAggregateHistory();
  return loadAggregate();
});
el('aggMultimodal').addEventListener('change', syncMultimodal);
syncMultimodal();

export { loadAggregate, renderAggregateResult, switchAggregatePane };
