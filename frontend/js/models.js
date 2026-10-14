import { el, api } from './core.js';

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
  if (el(`${prefix}HeadDetection`).checked) {
    el(wrapId).hidden = true;
    el(`${prefix}ModelHint`).hidden = false;
    el(`${prefix}ModelHint`).textContent = 'head检测：medium.pt ≥ 0.45 为基准，RT-DETRv4 ≥ 0.2 补齐；两路 imgsz 固定 640。仅加入与所有基准框 IoU 为 0 的人头框。';
    return;
  }
  if (prefix !== 'det') el(`${prefix}ModelHint`).hidden = true;
  const name = el(selId).value || '';
  el(wrapId).hidden = !WORLD_WEIGHTS.includes(name);
  if (selId === 'detWeights') {
    const headMode = name === RTDETR_STATUS?.weight;
    const hint = el('detModelHint');
    hint.hidden = !headMode;
    hint.textContent = headMode
      ? (RTDETR_STATUS.available ? '按人头统计人数；人体框不计入。可调大 imgsz 检测远处小人头。' : RTDETR_STATUS.message)
      : '';
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
    fillSelect('vidWeights', all, DEFAULT_WEIGHTS);   // 视频推理：全部权重
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
  return { weights, isWorld, classes, headDetection, isRtdetr: weights === RTDETR_STATUS?.weight };
}

const previousHeadOptions = new Map();
function syncHeadDetection(prefix) {
  const enabled = el(`${prefix}HeadDetection`).checked;
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
  el(`${prefix}HeadDetection`).addEventListener('change', () => syncHeadDetection(prefix));
});


export { loadEnv, readDetectOptions, DEFAULT_WEIGHTS };
