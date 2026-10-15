import { el, esc, api, showError } from './core.js?v=87';
import { bindImageHover, hideHoverPreview } from './preview.js?v=87';
import { classColor } from './viewer.js?v=87';

/* ==================== 聚合推理（目标检测 → 人脸识别 → 姿势检测绑定）==================== */

const STEP_LABELS = { detect: '目标检测', face: '人脸识别', pose: '姿势检测' };
/** 只画 head 框；筛选按「已贴姓名 / 未识别 / 已绑姿势」分档，与图片推理的类别胶囊同一套交互 */
const FILTERS = [
  { key: 'all', label: '全部 head' },
  { key: 'named', label: '已贴姓名' },
  { key: 'unknown', label: '未识别' },
  { key: 'posed', label: '已绑姿势' },
];
const AGG_PANES = { images: 'aggregate-pane-images', result: 'aggregate-pane-result' };
// 与后端 aggregate.MIN_FACE_CONFIDENCE / MIN_MATCH_THRESHOLD 同源：低于下限会被抬到下限
const FACE_CONFIDENCE_FLOOR = 0.5;
const MATCH_THRESHOLD_FLOOR = 0.3;
let ACTIVE_RECORD = null;
let ACTIVE_FILTER = 'all';
let ALWAYS_TAGS = false;

/** 读取带下限的参数：低于下限抬到下限（并回写输入框，界面显示的就是实际生效值），超上限报错。 */
function readFloored(id, floor, upper, label, fallback) {
  const value = Number(el(id).value || fallback);
  if (!Number.isFinite(value) || value > upper) throw new Error(`${label}必须在 ${floor} 到 ${upper} 之间`);
  const effective = Math.max(floor, value);
  el(id).value = String(effective);
  return effective;
}

/** 读取聚合参数：人脸识别 / 姿势检测由勾选框决定是否参与。
 *
 * 目标检测与绑定规则都不在这里配——后端固定（图片推理那套人头检测 + 链式贪心一一对应）。
 */
function readAggregateOptions() {
  return {
    face: !!el('aggFaceRecognition').checked, pose: !!el('aggPose').checked,
    threshold: readFloored('aggRecThreshold', MATCH_THRESHOLD_FLOOR, 1, '识别阈值', '0.4'),
    face_confidence: readFloored('aggFaceConfidence', FACE_CONFIDENCE_FLOOR, 0.99, '人脸置信度', '0.5'),
  };
}

/* ==================== 子标签页（待推理图片 / 聚合结果，布局与图片推理一致）==================== */

/** pane: 'images' | 'result' */
function switchAggregatePane(pane) {
  Object.entries(AGG_PANES).forEach(([key, id]) => el(id).classList.toggle('active', key === pane));
  document.querySelectorAll('.subtab[data-agg]').forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.agg === pane);
  });
  el('aggRefresh').hidden = pane !== 'images';
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
    if (!step || !step.enabled) return `<span class="agg-step is-off">${esc(label)}：未启用</span>`;
    const detail = key === 'detect' ? `${step.count} 个人头`
      : key === 'face' ? `检出 ${step.face_count} 张 · 识别 ${step.matched} · 库内 ${step.library_count} 人 · 阈值 ${step.threshold}`
        : `${step.count} 人 · 绑定 ${step.bound} · 未绑定 ${step.unbound}`;
    return `<span class="agg-step"><b>${esc(label)}</b>：${esc(detail)}｜${esc(step.elapsed_seconds)}s</span>`;
  }).join('');
}

/** head 框上的标签：姓名 + 相似度（未识别就写类别 + 置信度），再挂上姿势状态 */
function headLabel(target) {
  const states = [...new Set((target.poses || []).map((pose) => pose.behavior?.state).filter(Boolean))];
  const poseNote = states.length ? ` · ${states.join('、')}` : '';
  return target.name
    ? `${target.name} ${target.similarity}${poseNote}`
    : `${target.class_name} ${target.confidence}${poseNote}`;
}

function visibleTargets() {
  const targets = ACTIVE_RECORD?.targets || [];
  if (ACTIVE_FILTER === 'named') return targets.filter((target) => target.name);
  if (ACTIVE_FILTER === 'unknown') return targets.filter((target) => !target.name);
  if (ACTIVE_FILTER === 'posed') return targets.filter((target) => (target.poses || []).length);
  return targets;
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
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = headLabel(target);
    node.appendChild(tag);
    layer.appendChild(node);
  });
  box.querySelectorAll('[data-role="headrow"]').forEach((row) => {
    row.hidden = !shown.includes(record.targets[Number(row.dataset.i)]);
  });
  const counts = {
    all: record.targets.length,
    named: record.targets.filter((target) => target.name).length,
    unknown: record.targets.filter((target) => !target.name).length,
    posed: record.targets.filter((target) => (target.poses || []).length).length,
  };
  box.querySelectorAll('[data-role="aggfilter"] .chip').forEach((chip) => {
    chip.textContent = `${chip.dataset.label} ${counts[chip.dataset.filter]}`;
    chip.style.opacity = chip.dataset.filter === ACTIVE_FILTER ? '1' : '.5';
  });
  const countNode = box.querySelector('#aggResultCount');
  if (countNode) {
    countNode.textContent =
      `图上 ${shown.length} 个 head 框 / 共检出 ${record.targets.length} 个｜已贴姓名 ${counts.named} 个` +
      (record.summary ? `｜姿势绑定 ${record.summary.bound_poses}/${record.summary.poses}` : '');
  }
}

function renderAggregateResult(record, filename) {
  const box = el('aggResult');
  ACTIVE_RECORD = record;
  ACTIVE_FILTER = 'all';
  const summary = record.summary || {};
  el('aggResultName').textContent = filename;
  el('aggStatus').textContent =
    `人头 ${record.count} 个｜贴姓名 ${summary.named_targets ?? 0} 个｜` +
    `人脸 ${summary.faces ?? 0} 张（已识别 ${summary.matched_faces ?? 0}、未识别 ${summary.unknown_faces ?? 0}、质量不足 ${summary.rejected_faces ?? 0}）｜` +
    `姿势 ${summary.poses ?? 0} 人（已绑定 ${summary.bound_poses ?? 0}）｜总耗时 ${record.elapsed_seconds}s`;

  const notes = [];
  if ((record.unbound_faces || []).length) notes.push(`没落到 head 框的人脸：${record.unbound_faces.join('、')}`);
  if ((record.unbound_poses || []).length) notes.push(`没绑上 head 框的姿势：第 ${record.unbound_poses.join('、')} 人`);
  const rows = (record.targets || []).map((target, index) => {
    const [x1, y1, x2, y2] = target.bbox;
    const poses = (target.poses || []).map((pose) =>
      `${esc(pose.behavior?.state || '无法判断')} <small>IoU ${esc(pose.bind?.iou)}</small>`).join('、');
    return `<tr data-role="headrow" data-i="${index}">` +
      `<td>${esc(target.class_name)} <small>${esc(target.confidence)}</small></td>` +
      `<td>${target.name ? `<b class="agg-name">${esc(target.name)}</b> <small>${esc(target.similarity)}</small>` : '<span class="agg-muted">—</span>'}</td>` +
      `<td>${poses || '<span class="agg-muted">—</span>'}</td>` +
      `<td>${esc(Math.round((x1 + x2) / 2))}, ${esc(Math.round((y1 + y2) / 2))}</td>` +
      `<td>${esc(x1)}, ${esc(y1)}, ${esc(x2)}, ${esc(y2)}</td></tr>`;
  }).join('');

  box.innerHTML =
    `<div class="agg-steps">${stepLine(record.steps)}</div>` +
    (notes.length ? `<p class="hint">${esc(notes.join('；'))}</p>` : '') +
    `<div class="viewer-bar">` +
    `<label><input type="checkbox" data-role="alwaystag"> 常显标签</label>` +
    `<div class="chips" data-role="aggfilter">` +
    FILTERS.map((filter, index) => `<button class="chip" data-filter="${filter.key}" data-label="${esc(filter.label)}"` +
      ` style="background:${classColor(index)}">${esc(filter.label)}</button>`).join('') +
    `</div></div>` +
    `<div class="viewer" data-role="viewer">` +
    `<img class="base" data-role="baseimg" src="/images/${encodeURIComponent(filename)}" alt="聚合图">` +
    `<div class="boxes" data-role="boxes"></div></div>` +
    `<div class="viewer-foot"><span id="aggResultCount"></span>` +
    (record.result_url ? `<a href="${esc(record.result_url)}" target="_blank">后端渲染图（对照）</a>` : '') +
    `</div>` +
    `<div class="det-table-wrap"><table><thead><tr><th>人头</th><th>姓名</th><th>姿势（绑定）</th>` +
    `<th>中心 (cx, cy)</th><th>左上右下 (x1, y1, x2, y2)</th></tr></thead><tbody>${rows}</tbody></table></div>` +
    `<p class="hint">目标检测：${esc(record.steps?.detect?.rule || '')}<br>` +
    `姓名归属与姿势绑定：${esc(record.bind?.rule || '')}。结果只画 head 框，姓名与姿势状态都挂在这个框上；` +
    `没绑上的人脸 / 姿势在清单里列出来，不会硬塞给某个人头。</p>`;

  const tagToggle = box.querySelector('[data-role="alwaystag"]');
  tagToggle.checked = ALWAYS_TAGS;
  tagToggle.addEventListener('change', event => {
    ALWAYS_TAGS = event.target.checked;
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

async function aggregateImage(filename) {
  const box = el('aggResult');
  ACTIVE_RECORD = null;
  switchAggregatePane('result');
  el('aggResultName').textContent = filename;
  el('aggStatus').textContent = '';
  const t0 = Date.now();
  let ticker = null;
  try {
    const payload = readAggregateOptions();
    const paint = () => {
      const seconds = ((Date.now() - t0) / 1000).toFixed(1);
      box.innerHTML = `<div>聚合推理中… <b>${seconds}s</b></div>` +
        `<div style="font-size:12px;color:#80868b;margin-top:6px">图片：${esc(filename)}｜` +
        `人脸识别 ${payload.face ? '开' : '关'}｜姿势检测 ${payload.pose ? '开' : '关'}<br>` +
        `（目标检测固定用「图片推理 → 目标检测」；第一次运行要加载模型，超过 600 秒自动报超时）</div>`;
    };
    paint();
    ticker = setInterval(paint, 200);
    const result = await api(`/api/aggregate/image/${encodeURIComponent(filename)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    }, 600000);
    if (ticker !== null) { clearInterval(ticker); ticker = null; }
    if (!result || !Array.isArray(result.targets)) {
      throw new Error(`后端返回数据不完整：${JSON.stringify(result).slice(0, 200)}`);
    }
    renderAggregateResult(result, filename);
  } catch (error) {
    if (ticker !== null) { clearInterval(ticker); ticker = null; }
    showError(box, error, '聚合失败');
  }
}

async function loadAggregate() {
  const grid = el('aggImageGrid');
  try {
    const data = await api('/api/images');
    hideHoverPreview();
    if (!data.images.length) {
      grid.innerHTML = '<div class="empty">暂无图片，请先上传</div>';
      return;
    }
    grid.innerHTML = '';
    for (const image of data.images) {
      const figure = document.createElement('figure');
      figure.title = `点击聚合：${image.filename}`;
      figure.innerHTML = `<img src="${esc(image.url)}" alt=""><figcaption>${esc(image.filename)}</figcaption>`;
      bindImageHover(figure, () => ({ src: image.url, name: image.filename }));
      figure.addEventListener('click', () => { hideHoverPreview(); aggregateImage(image.filename); });
      grid.appendChild(figure);
    }
  } catch (error) {
    grid.innerHTML = `<div class="empty">加载失败：${esc(error instanceof Error ? error.message : String(error))}</div>`;
  }
}

el('aggRefresh').addEventListener('click', loadAggregate);
el('aggFaceRecognition').addEventListener('change', () => { el('aggRecThreshold').disabled = !el('aggFaceRecognition').checked; });
el('aggPose').addEventListener('change', () => { el('aggPoseConfidence').disabled = !el('aggPose').checked; });

export { loadAggregate, aggregateImage, renderAggregateResult, switchAggregatePane };
