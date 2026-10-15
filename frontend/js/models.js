import { el, api } from './core.js?v=71';

/* ==================== 环境状态 ==================== */

/* ==================== 环境状态 / 模型模式（普通 YOLO ↔ YOLO-World）==================== */

let WORLD_WEIGHTS = [];   // 属于 YOLO-World 的权重文件名（由后端告知）
let DEFAULT_WEIGHTS = 'medium.pt';   // 下拉框默认选中（后端 status 覆盖）
let MODEL_LABELS = {};
let RTDETR_STATUS = null;

/** 填充下拉框，并尽量保留原选择；无选择时用 preferred（默认权重） */
function fillSelect(id, names, preferred = '') {
  const sel = el(id);
  const cur = sel.value;
  sel.innerHTML = '';
  if (!names.length) {
    sel.innerHTML = '<option value="">（无可用权重）</option>';
    return;
  }
  names.forEach((m) => {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = MODEL_LABELS[m] || m;
    sel.appendChild(o);
  });
  if (cur && names.includes(cur)) sel.value = cur;
  else if (preferred && names.includes(preferred)) sel.value = preferred;
}

/** 只有 World 权重才需要填 prompt，普通 YOLO 的类别是权重内置的，直接隐藏输入框 */
function syncModeUi(selId, wrapId) {
  const prefix = selId.replace('Weights', '');
  el(`${prefix}PoseOptions`).hidden = el(`${prefix}HeadDetection`).checked || !el(selId).value.toLowerCase().includes('pose');
  if (el(`${prefix}HeadDetection`).checked) {
    el(wrapId).hidden = true;
    el(`${prefix}ModelHint`).hidden = false;
    el(`${prefix}ModelHint`).textContent = el(`${prefix}RtdetrBase`).checked
      ? `目标检测：RT-DETRv4 ≥ 0.25 为基准，medium.pt ≥ 0.45 补齐；两路 imgsz 固定 640。IoU 超过 ${el(`${prefix}MergeIou`).value || 0} 保留 RT-DETR，否则补入 medium。`
      : `目标检测：medium.pt ≥ 0.45 为基准，RT-DETRv4 ≥ 0.25 补齐；两路 imgsz 固定 640。IoU 超过 ${el(`${prefix}MergeIou`).value || 0} 保留 medium，否则补入 RT-DETR。`;
    el(`${prefix}ModelHint`).textContent += ` 仅待补入框置信度 > ${el(`${prefix}MergeConfidence`).value || 0} 时使用自定义 IoU，其余仍按 IoU=0。`;
    return;
  }
  if (prefix !== 'det') el(`${prefix}ModelHint`).hidden = true;
  const name = el(selId).value || '';
  el(wrapId).hidden = !WORLD_WEIGHTS.includes(name);
  if (selId === 'detWeights' || selId === 'vidWeights') {
    const headMode = name === RTDETR_STATUS?.weight;
    const hint = el(`${prefix}ModelHint`);
    hint.hidden = !headMode;
    hint.textContent = headMode
      ? (RTDETR_STATUS.available ? (prefix === 'vid' ? 'RT-DETRv4 按人头统计人数，输入尺寸 640。' : '按人头统计人数；人体框不计入。可调大 imgsz 检测远处小人头。') : RTDETR_STATUS.message)
      : '';
    if (name.toLowerCase().includes('pose')) {
      hint.hidden = false;
      hint.textContent = 'Pose：人体框 + 上半身 11 点骨架，显示关键点置信度 ≥ 0.5 的连线。原始 17 个关键点完整保存。';
    }
  }
}

async function loadEnv() {
  const badge = el('envStatus');
  try {
    const s = await api('/api/yolo/status');
    badge.textContent = s.message;
    badge.className = 'badge ' + (s.available ? 'ok' : 'err');

    WORLD_WEIGHTS = s.world_weights || [];
    DEFAULT_WEIGHTS = s.default_weights || DEFAULT_WEIGHTS;
    MODEL_LABELS = s.model_labels || {};
    RTDETR_STATUS = s.rtdetrv4 || null;

    const all = s.weights.length ? s.weights : [DEFAULT_WEIGHTS];
    fillSelect('detWeights', s.image_weights || all, s.default_image_weights || DEFAULT_WEIGHTS);
    fillSelect('vidWeights', s.video_weights || all, DEFAULT_WEIGHTS);
    fillSelect('trainWeights', all, DEFAULT_WEIGHTS); // 训练：全部权重（含 World，迁移起点与推理一致）
    fillSelect('annWeights', all, DEFAULT_WEIGHTS);   // 标注器：全部权重（与推理一致）

    syncModeUi('detWeights', 'detClassesWrap');
    syncModeUi('vidWeights', 'vidClassesWrap');
    syncModeUi('annWeights', 'annClassesWrap');
    ['det', 'vid', 'ann'].forEach((prefix) => {
      if (el(`${prefix}HeadDetection`).checked) syncHeadDetection(prefix);
    });
  } catch (e) {
    badge.textContent = '状态获取失败：' + (e instanceof Error ? e.message : String(e));
    badge.className = 'badge err';
  }
}

// 切换权重 → 立即更新 prompt 输入框的显隐
el('detWeights').addEventListener('change', () =>
  syncModeUi('detWeights', 'detClassesWrap')
);
el('vidWeights').addEventListener('change', () =>
  syncModeUi('vidWeights', 'vidClassesWrap')
);
el('annWeights').addEventListener('change', () =>
  syncModeUi('annWeights', 'annClassesWrap')
);

/** 读取当前推理设置（图片/视频共用） */
function readDetectOptions(prefix) {
  const headDetection = !!el(`${prefix}HeadDetection`).checked;
  const weights = headDetection ? 'medium.pt' : el(`${prefix}Weights`).value || DEFAULT_WEIGHTS;
  const isWorld = WORLD_WEIGHTS.includes(weights);
  const classes = !headDetection && isWorld ? el(`${prefix}Classes`).value.trim() : '';
  const mergeIou = headDetection ? Number(el(`${prefix}MergeIou`).value || '0.1') : 0;
  const mergeConfidence = headDetection ? Number(el(`${prefix}MergeConfidence`).value || '0.55') : 0;
  const headDownRThreshold = Number(el(`${prefix}HeadDownR`).value || '0.05');
  const headUpRThreshold = Number(el(`${prefix}HeadUpR`).value || '0.20');
  const poseRoi = !!el(`${prefix}PoseRoi`).checked;
  const roiConfidence = Number(el(`${prefix}RoiConfidence`).value || '0.6');
  if (!Number.isFinite(roiConfidence) || roiConfidence < 0 || roiConfidence > 1) throw new Error('物体置信度必须在 0 到 1 之间');
  if ([headDownRThreshold, headUpRThreshold].some(v => !Number.isFinite(v) || v < -1 || v > 1)) throw new Error('低头和抬头 R 阈值必须在 -1 到 1 之间');
  if (headDownRThreshold >= headUpRThreshold) throw new Error('低头 R 阈值必须小于抬头 R 阈值');
  if (!Number.isFinite(mergeIou) || mergeIou < 0 || mergeIou > 1) throw new Error('合并 IoU 阈值必须在 0 到 1 之间');
  if (!Number.isFinite(mergeConfidence) || mergeConfidence < 0 || mergeConfidence > 1) throw new Error('自定义 IoU 的置信度门槛必须在 0 到 1 之间');
  return { weights, isWorld, classes, headDetection, mergeIou, mergeConfidence, headDownRThreshold, headUpRThreshold, poseRoi, roiConfidence, isPose: weights.toLowerCase().includes('pose'), rtdetrBase: headDetection && !!el(`${prefix}RtdetrBase`).checked, isRtdetr: weights === RTDETR_STATUS?.weight };
}

const previousHeadOptions = new Map();
function syncHeadDetection(prefix) {
  const enabled = el(`${prefix}HeadDetection`).checked;
  el(`${prefix}RtdetrBase`).disabled = !enabled;
  el(`${prefix}MergeIou`).disabled = !enabled;
  el(`${prefix}MergeConfidence`).disabled = !enabled;
  const ids = [`${prefix}Weights`, `${prefix}Conf`, ...(prefix === 'det' ? ['detImgsz'] : [])];
  if (enabled) {
    if (!previousHeadOptions.has(prefix)) previousHeadOptions.set(prefix, ids.map((id) => el(id).value));
    ids.forEach((id, index) => { el(id).value = ['medium.pt', '0.45', '640'][index]; });
  } else if (previousHeadOptions.has(prefix)) {
    ids.forEach((id, index) => { el(id).value = previousHeadOptions.get(prefix)[index]; });
    previousHeadOptions.delete(prefix);
  }
  ids.forEach((id) => { el(id).disabled = enabled; });
  syncModeUi(`${prefix}Weights`, `${prefix}ClassesWrap`);
}
['det', 'vid', 'ann'].forEach((prefix) => {
  ['Down', 'Up'].forEach(direction => {
    const slider = el(`${prefix}Head${direction}RSlider`), number = el(`${prefix}Head${direction}R`);
    slider.addEventListener('input', () => { number.value = slider.value; });
    number.addEventListener('input', () => {
      const value = Number(number.value);
      if (number.value !== '' && Number.isFinite(value) && value >= -1 && value <= 1) slider.value = number.value;
    });
  });
  el(`${prefix}HeadDetection`).addEventListener('change', () => syncHeadDetection(prefix));
  el(`${prefix}RtdetrBase`).addEventListener('change', () => syncModeUi(`${prefix}Weights`, `${prefix}ClassesWrap`));
  el(`${prefix}MergeIou`).addEventListener('input', () => syncModeUi(`${prefix}Weights`, `${prefix}ClassesWrap`));
  el(`${prefix}MergeConfidence`).addEventListener('input', () => syncModeUi(`${prefix}Weights`, `${prefix}ClassesWrap`));
});


export { loadEnv, readDetectOptions, DEFAULT_WEIGHTS };
