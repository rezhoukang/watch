import { el, esc, api, showError } from './core.js?v=83';
import { bindImageHover, hideHoverPreview } from './preview.js?v=83';

/* ==================== 聚合操作（目标检测 → 人脸识别 → 姿势检测绑定）==================== */

const STEP_LABELS = { detect: '目标检测', face: '人脸识别', pose: '姿势检测' };
const FACE_STATUS = { matched: '已识别', unknown: '未识别', rejected: '质量不足', error: '识别失败', pending: '待比对' };

/** 读取聚合参数：人脸识别 / 姿势检测由勾选框决定是否参与 */
function readAggregateOptions() {
  const weights = el('aggWeights').value || 'medium.pt';
  const confidence = Number(el('aggConf').value || '0.38');
  if (!Number.isFinite(confidence) || confidence < 0.01 || confidence > 0.99) throw new Error('置信度必须在 0.01 到 0.99 之间');
  const imgsz = Number(el('aggImgsz').value || '640');
  if (!Number.isInteger(imgsz) || imgsz < 320 || imgsz > 1920) throw new Error('尺寸 imgsz 必须是 320 到 1920 之间的整数');
  const threshold = Number(el('aggRecThreshold').value || '0.4');
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('识别阈值必须在 0 到 1 之间');
  const faceConfidence = Number(el('aggFaceConfidence').value || '0.5');
  if (!Number.isFinite(faceConfidence) || faceConfidence < 0.01 || faceConfidence > 0.99) throw new Error('人脸置信度必须在 0.01 到 0.99 之间');
  const poseConfidence = Number(el('aggPoseConfidence').value || '0.6');
  if (!Number.isFinite(poseConfidence) || poseConfidence < 0.01 || poseConfidence > 0.99) throw new Error('姿势置信度必须在 0.01 到 0.99 之间');
  const bindIou = Number(el('aggBindIou').value || '0.3');
  if (!Number.isFinite(bindIou) || bindIou < 0 || bindIou > 1) throw new Error('绑定 IoU 阈值必须在 0 到 1 之间');
  return {
    weights, confidence, imgsz, classes: el('aggClasses').value.trim(),
    face: !!el('aggFaceRecognition').checked, pose: !!el('aggPose').checked,
    threshold, face_confidence: faceConfidence, pose_confidence: poseConfidence,
    bind: el('aggBind').value, bind_iou: bindIou,
  };
}

/** 三步各自的耗时与统计（未勾选的显示「未启用」，避免看起来像漏跑） */
function stepLine(steps) {
  return Object.entries(STEP_LABELS).map(([key, label]) => {
    const step = (steps || {})[key];
    if (!step || !step.enabled) return `<span class="agg-step is-off">${esc(label)}：未启用</span>`;
    const detail = key === 'detect' ? `${step.count} 个目标 · ${step.weight} · ${step.confidence}`
      : key === 'face' ? `检出 ${step.face_count} 张 · 识别 ${step.matched} · 库内 ${step.library_count} 人 · 阈值 ${step.threshold}`
      : `${step.count} 人 · 绑定 ${step.bound} · ${step.strategy} ≥ ${step.iou_threshold}`;
    return `<span class="agg-step"><b>${esc(label)}</b>：${esc(detail)}｜${esc(step.elapsed_seconds)}s</span>`;
  }).join('');
}

function cell(value, fallback = '—') {
  return value ? esc(value) : `<span class="agg-muted">${esc(fallback)}</span>`;
}

/** 目标明细：识别到的姓名、对应人脸、绑定上的姿势 */
function targetRow(target, index) {
  const name = target.name
    ? `<b class="agg-name">${esc(target.name)}</b> <small>${esc(target.similarity)}</small>`
    : cell('');
  const face = target.face_id ? `<small>#${esc(target.face_id)}</small>` : cell('');
  const poses = (target.poses || []).map((pose) =>
    `${esc(pose.behavior?.state || '无法判断')}<small>（IoU ${esc(pose.bind?.iou)} / ${esc(pose.bind?.strategy || '')}）</small>`).join('、');
  return `<tr><td>${index + 1}</td><td>${esc(target.class_name)} <small>${esc(target.confidence)}</small></td>
    <td>${name}</td><td>${face}</td><td>${poses || cell('')}</td>
    <td><small>${esc((target.bbox || []).join(', '))}</small></td></tr>`;
}

function renderAggregateResult(r, filename) {
  const box = el('aggResult');
  const summary = r.summary || {};
  el('aggResultName').textContent = filename;
  el('aggStatus').textContent =
    `目标 ${r.count} 个｜贴姓名 ${summary.named_targets ?? 0} 个｜` +
    `人脸 ${summary.faces ?? 0} 张（已识别 ${summary.matched_faces ?? 0}、未识别 ${summary.unknown_faces ?? 0}、质量不足 ${summary.rejected_faces ?? 0}）｜` +
    `姿势 ${summary.poses ?? 0} 人（已绑定 ${summary.bound_poses ?? 0}）｜总耗时 ${r.elapsed_seconds}s`;

  const notes = [];
  if ((r.unbound_faces || []).length) notes.push(`没落到任何目标框的人脸：${r.unbound_faces.join('、')}`);
  if ((r.unbound_poses || []).length) notes.push(`没绑上目标的姿势：第 ${r.unbound_poses.join('、')} 人`);
  const rows = (r.targets || []).map(targetRow).join('');
  box.innerHTML =
    `<div class="agg-steps">${stepLine(r.steps)}</div>` +
    (notes.length ? `<p class="hint">${esc(notes.join('；'))}</p>` : '') +
    `<figure class="agg-figure"><img src="${esc(r.result_url)}" alt="${esc(filename)} 聚合结果">` +
    `<figcaption>${esc(r.result || '')}</figcaption></figure>` +
    `<table class="agg-table"><thead><tr><th>#</th><th>目标</th><th>姓名</th><th>人脸</th><th>姿势（绑定）</th><th>坐标</th></tr></thead>` +
    `<tbody>${rows || '<tr><td colspan="6">没有检出目标</td></tr>'}</tbody></table>` +
    `<p class="hint">姓名归属：人脸中心落在哪个目标框内就归给谁（多个框都包含时取最小的）；都不包含时吸附到最近目标中心。` +
    `姿势绑定规则：${esc(r.bind?.rule || '')}。人脸框颜色：绿=已识别、橙=未识别、灰=质量不足。</p>`;
}

async function aggregateImage(filename) {
  const box = el('aggResult');
  el('aggResultName').textContent = filename;
  el('aggStatus').textContent = '';
  const t0 = Date.now();
  let ticker = null;
  try {
    const payload = readAggregateOptions();
    const paint = () => {
      const seconds = ((Date.now() - t0) / 1000).toFixed(1);
      box.innerHTML = `<div>聚合中… <b>${seconds}s</b></div>` +
        `<div style="font-size:12px;color:#80868b;margin-top:6px">图片：${esc(filename)}｜目标检测 ${esc(payload.weights)}｜` +
        `人脸识别 ${payload.face ? '开' : '关'}｜姿势检测 ${payload.pose ? '开' : '关'}<br>` +
        `（第一次运行要加载模型，可能较慢；超过 600 秒自动报超时）</div>`;
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
    hidePreview();
    if (!data.images.length) {
      grid.innerHTML = '<div class="empty">暂无待推理图片，请先在「上传」页上传</div>';
      return;
    }
    grid.innerHTML = '';
    for (const image of data.images) {
      const figure = document.createElement('figure');
      figure.title = `点击聚合：${image.filename}`;
      figure.innerHTML = `<img src="${esc(image.url)}" alt="" loading="lazy"><figcaption>${esc(image.filename)}</figcaption>`;
      bindImageHover(figure, () => ({ src: image.url, name: image.filename }));
      figure.addEventListener('click', () => { hidePreview(); aggregateImage(image.filename); });
      grid.appendChild(figure);
    }
  } catch (error) {
    grid.innerHTML = `<div class="empty">加载失败：${esc(error instanceof Error ? error.message : String(error))}</div>`;
  }
}

el('aggRefresh').addEventListener('click', loadAggregate);

export { loadAggregate, aggregateImage, renderAggregateResult };
