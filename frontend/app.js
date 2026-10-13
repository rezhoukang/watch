/* 本地视觉训练平台 - 前端逻辑（纯 JavaScript，无需编译）
 *
 * 直接改本文件 → 刷新浏览器生效。
 * 注意：改完请把 index.html 里引用的 ?v=N 加 1（清浏览器缓存）。
 */

/* ==================== 通用工具 ==================== */

/** 取元素，取不到就抛明确错误（避免"点了没反应"） */
function el(id) {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`页面元素缺失：#${id}（请强制刷新页面 Ctrl+F5）`);
  }
  return node;
}

/** HTML 转义，避免内容破坏页面结构 */
function esc(v) {
  return String(v == null ? '' : v).replace(/[&<>"']/g, (c) => {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return map[c];
  });
}

/** 把错误显示到指定区域（同时打印完整堆栈到控制台） */
function showError(box, err, prefix = '失败') {
  const e = err instanceof Error ? err : new Error(String(err));
  console.error(`[${prefix}]`, e);
  if (box) {
    box.innerHTML =
      `<div style="color:#c5221f;font-weight:600">${esc(prefix)}：${esc(e.message)}</div>` +
      `<div style="font-size:12px;color:#80868b;margin-top:6px">` +
      `错误类型：${esc(e.name)}｜详细堆栈见浏览器控制台（F12 → Console）</div>`;
  }
}

/** 顶部红条：捕获脚本级异常，避免静默失败 */
function banner(msg) {
  let node = document.getElementById('globalBanner');
  if (!node) {
    node = document.createElement('div');
    node.id = 'globalBanner';
    node.style.cssText =
      'position:fixed;top:0;left:0;right:0;z-index:99;background:#c5221f;color:#fff;' +
      'padding:8px 14px;font-size:13px;font-family:inherit';
    document.body.appendChild(node);
  }
  node.textContent = msg;
}

window.addEventListener('error', (ev) => {
  banner(`脚本异常：${ev.message}（${ev.filename}:${ev.lineno}）`);
});

window.addEventListener('unhandledrejection', (ev) => {
  const r = ev.reason;
  banner(`异步异常：${r instanceof Error ? r.message : String(r)}`);
});

/* 禁用浏览器对图片的「原生拖拽」
 *
 * 标注是「长按拖拽画框」的手势，而浏览器在按住图片拖动时会启动
 * 自己的拖拽（图片幽灵跟着光标、松手后可能触发导航），两者直接冲突。
 * 所以：
 *   1. 全局拦 dragstart（CSS 的 -webkit-user-drag 只管 Chromium，
 *      Firefox 不认，必须在 JS 里拦才跨浏览器）
 *   2. 标注器里的图片再设 draggable=false（掐掉可能发起拖拽的源头）
 */
document.addEventListener('dragstart', (ev) => {
  const t = ev.target;
  if (!t) return;
  // 图片本身，或标注器里的任何元素，都不允许发起原生拖拽
  if (t.tagName === 'IMG' || (t.closest && t.closest('#annotatorMask'))) {
    ev.preventDefault();
  }
});
el('annBaseImg').draggable = false;

/* ==================== 请求封装（带超时 + 明确错误）==================== */

async function api(path, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const t0 = Date.now();

  try {
    const res = await fetch(path, { ...options, signal: controller.signal });
    const text = await res.text();

    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (e) {
        data = null;
      }
    }

    if (!res.ok) {
      const detail = (data && (data.detail || data.message)) || text.slice(0, 200) || '(空响应体)';
      throw new Error(`HTTP ${res.status} ${res.statusText} — ${detail}`);
    }
    if (data === null) {
      throw new Error(`响应不是合法 JSON（HTTP ${res.status}）：${text.slice(0, 200)}`);
    }
    return data;
  } catch (err) {
    // 超时
    if (err && err.name === 'AbortError') {
      throw new Error(`请求超时：${timeoutMs / 1000} 秒内无响应 — ${path}`);
    }
    // 连不上后端
    if (err instanceof TypeError) {
      throw new Error(`无法连接后端（服务未启动 / 已关闭 / 端口不对？）：${err.message}`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
    console.debug(`[api] ${path} 用时 ${Date.now() - t0}ms`);
  }
}

/* ==================== 标签页切换 ==================== */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');

    const name = tab.dataset.tab || '';
    el(`tab-${name}`).classList.add('active');

    if (name !== 'detect') hideHoverPreview();

    if (name === 'detect') loadImages();
    if (name === 'video') loadVideos();
    // 训练页只要数据集下拉框；数据集管理页还要标注缩略图
    if (name === 'train') loadDatasets();
    if (name === 'dataset') loadDatasets().then(loadAnnotateGrid);
  });
});

/* ==================== 环境状态 ==================== */

/* ==================== 环境状态 / 模型模式（普通 YOLO ↔ YOLO-World）==================== */

let WORLD_WEIGHTS = [];   // 属于 YOLO-World 的权重文件名（由后端告知）
let DEFAULT_WEIGHTS = 'medium.pt';   // 下拉框默认选中（后端 status 覆盖）

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
    o.textContent = m;
    sel.appendChild(o);
  });
  if (cur && names.includes(cur)) sel.value = cur;
  else if (preferred && names.includes(preferred)) sel.value = preferred;
}

/** 只有 World 权重才需要填 prompt，普通 YOLO 的类别是权重内置的，直接隐藏输入框 */
function syncModeUi(selId, wrapId) {
  const name = el(selId).value || '';
  el(wrapId).hidden = !WORLD_WEIGHTS.includes(name);
}

async function loadEnv() {
  const badge = el('envStatus');
  try {
    const s = await api('/api/yolo/status');
    badge.textContent = s.message;
    badge.className = 'badge ' + (s.available ? 'ok' : 'err');

    WORLD_WEIGHTS = s.world_weights || [];
    DEFAULT_WEIGHTS = s.default_weights || DEFAULT_WEIGHTS;

    const all = s.weights.length ? s.weights : [DEFAULT_WEIGHTS];
    fillSelect('detWeights', all, DEFAULT_WEIGHTS);   // 图片推理：全部权重
    fillSelect('vidWeights', all, DEFAULT_WEIGHTS);   // 视频推理：全部权重
    fillSelect('trainWeights', all, DEFAULT_WEIGHTS); // 训练：全部权重（含 World，迁移起点与推理一致）
    fillSelect('annWeights', all, DEFAULT_WEIGHTS);   // 标注器：全部权重（与推理一致）

    syncModeUi('detWeights', 'detClassesWrap');
    syncModeUi('vidWeights', 'vidClassesWrap');
    syncModeUi('annWeights', 'annClassesWrap');
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
  const weights = el(`${prefix}Weights`).value || DEFAULT_WEIGHTS;
  const isWorld = WORLD_WEIGHTS.includes(weights);
  const classes = isWorld ? el(`${prefix}Classes`).value.trim() : '';
  return { weights, isWorld, classes };
}

/* ==================== 上传（图片 / 视频）==================== */

/**
 * kind: 'image' | 'video'
 */
function bindDropZone(zoneId, inputId, btnId, statusId, kind, uploadPath, onDone) {
  const zone = el(zoneId);
  const fileInput = el(inputId);
  const btn = el(btnId);
  const status = el(statusId);
  let files = [];

  const update = () => {
    const n = files.length;
    btn.disabled = n === 0;
    btn.textContent = n > 0 ? `上传 ${n} 个文件` : kind === 'image' ? '上传图片' : '上传视频';
    if (n > 0) {
      status.className = 'status';
      status.textContent = `已选择 ${n} 个文件`;
    }
  };

  fileInput.addEventListener('change', () => {
    files = Array.from(fileInput.files || []);
    update();
  });

  zone.addEventListener('click', () => fileInput.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const list = Array.from((e.dataTransfer && e.dataTransfer.files) || []);
    files = list.filter((f) => f.type.startsWith(`${kind}/`));
    update();
  });

  btn.addEventListener('click', async () => {
    if (!files.length) return;
    btn.disabled = true;
    status.className = 'status';
    status.textContent = '上传中…';

    const ok = [];
    const fail = [];

    try {
      for (const f of files) {
        const fd = new FormData();
        fd.append('file', f);
        try {
          const d = await api(uploadPath, { method: 'POST', body: fd }, 120000);
          ok.push(d && d.overwritten ? `${f.name}（已覆盖同名文件）` : f.name);
        } catch (e) {
          fail.push(`${f.name}: ${e instanceof Error ? e.message : String(e)}`);
        }
      }

      status.className = 'status' + (fail.length ? ' error' : '');
      status.textContent = `成功 ${ok.length} 个${fail.length ? `，失败 ${fail.length} 个：${fail.join('；')}` : ''}`;
    } finally {
      files = [];
      fileInput.value = '';
      update();
      if (onDone) onDone();
    }
  });
}

bindDropZone('dropZone', 'fileInput', 'uploadBtn', 'status', 'image', '/api/upload', loadImages);
bindDropZone('dropZoneVideo', 'videoInput', 'uploadVideoBtn', 'videoStatus', 'video', '/api/upload_video', loadVideos);

/* ==================== 缩略图悬浮大图预览 ==================== */

let hoverEv = null;

/** 按鼠标位置摆放预览框；贴到右/下边缘时自动翻到另一侧 */
function placeHoverPreview(ev) {
  const box = el('hoverPreview');
  const pad = 18;
  const rect = box.getBoundingClientRect();

  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + rect.width > window.innerWidth - 10) x = ev.clientX - pad - rect.width;
  if (y + rect.height > window.innerHeight - 10) y = window.innerHeight - 10 - rect.height;
  if (x < 10) x = 10;
  if (y < 10) y = 10;

  box.style.left = `${Math.round(x)}px`;
  box.style.top = `${Math.round(y)}px`;
}

function showHoverPreview(src, name, ev) {
  const box = el('hoverPreview');
  const img = box.querySelector('img');
  hoverEv = ev;
  if (img.getAttribute('src') !== src) img.src = src;
  el('hoverPreviewCap').textContent = name || '';
  box.style.display = 'block';
  placeHoverPreview(ev);
}

function hideHoverPreview() {
  hoverEv = null;
  const box = document.getElementById('hoverPreview');
  if (box) box.style.display = 'none';
}

// 图片下载完才知道真实尺寸，加载完再定位一次
el('hoverPreview').querySelector('img').addEventListener('load', () => {
  if (hoverEv) placeHoverPreview(hoverEv);
});

/* ==================== 图片推理：图库 / 结果 两个子页面 ==================== */

const SUBTAB_PANES = { images: 'detect-pane-images', result: 'detect-pane-result' };

/** pane: 'images' | 'result' */
function switchSubPane(pane) {
  Object.entries(SUBTAB_PANES).forEach(([key, id]) => {
    el(id).classList.toggle('active', key === pane);
  });
  document.querySelectorAll('.subtab').forEach((t) => {
    t.classList.toggle('active', t.dataset.sub === pane);
  });
  el('refreshImages').hidden = pane !== 'images';
  if (pane !== 'images') hideHoverPreview();
  window.scrollTo(0, 0);
}

document.querySelectorAll('.subtab').forEach((t) => {
  t.addEventListener('click', () => switchSubPane(t.dataset.sub || 'images'));
});

/* ==================== 推理结果查看器（框由前端渲染）==================== */

/** 去重默认开关：勾选框与 view.dedup 的初值都取它，避免「界面勾着但没生效」 */
const DEDUP_DEFAULT = false;

/** 类别 → 颜色：黄金角取色，区分度高且白字可读 */
function classColor(i) {
  return `hsl(${Math.round((i * 137.508) % 360)}, 62%, 42%)`;
}

/** 同上，但带透明度：给「框半透明、中心点不透明」用 */
function classColorA(i, alpha) {
  return `hsla(${Math.round((i * 137.508) % 360)}, 62%, 42%, ${alpha})`;
}

/**
 * 渲染推理结果查看器（原图 + 前端叠加框 + 筛选条 + 明细表）
 * 图片页与视频页「图预览」共用；opts.root 指定渲染到哪个容器。
 *
 * @param {object} r      推理结果 {detections, mode, weight, classes_used, imgsz, ...}
 * @param {string} filename 标题（图片文件名 / 视频名+帧信息）
 * @param {number} confUsed 置信度初值
 * @param {number} elapsed  推理耗时（秒）
 * @param {number} imgsz    推理尺寸
 * @param {object} opts    { root, imgUrl }
 */
function renderDetectResult(r, filename, confUsed, elapsed, imgsz, opts = {}) {
  const box = opts.root || el('detectResult');
  const imgUrl = opts.imgUrl || `/images/${encodeURIComponent(filename)}`;
  const dets = Array.isArray(r.detections) ? r.detections : [];

  // 类别按出现次数降序，分配固定颜色
  const counts = new Map();
  dets.forEach((d) => counts.set(d.class_name, (counts.get(d.class_name) || 0) + 1));
  const classes = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([name]) => name);
  const colorOf = new Map(classes.map((name, i) => [name, classColor(i)]));

  // 作用域状态（不再用全局 DET，支持多个查看器共存）
  const view = {
    root: box,
    dets,
    classes,
    colorOf,
    hidden: new Set(),          // 被关掉的类别
    solo: null,                 // 双击胶囊 → 只看这一类
    shape: 'box',               // 图形：'box' 框 | 'fill' 实心 | 'dot' 中心点
    confMin: Number(confUsed) || 0.01,
    naturalW: 1,
    naturalH: 1,
    // 去重（后处理，不重跑模型）
    dedup: DEDUP_DEFAULT,
    dedupDist: 20,              // 中心点距离 ≤ 这个像素数 → 视为同一目标
    dedupNest: 0.8,             // 一个框被另一个覆盖 ≥ 这个比例 → 视为同一目标
    dedupKeep: 'conf',          // 冲突时先留谁：'conf' 得分高 / 'big' 大框 / 'small' 小框
  };
  view.q = (sel) => box.querySelector(sel);
  view.qs = (sel) => [...box.querySelectorAll(sel)];

  const modeLabel = r.mode === 'world' ? 'YOLO-World' : '普通 YOLO';
  const clsNote =
    r.mode === 'world'
      ? `prompt「${(r.classes_used || []).join(', ') || '权重内置'}」`
      : `内置 ${(r.classes_used || []).length} 类`;

  box.innerHTML =
    `<div class="result-head">${esc(filename)}` +
      `<span class="meta">${esc(modeLabel)}｜${esc(r.weight || '')}｜${esc(clsNote)}｜` +
      `推理阈值 ${esc(confUsed)}｜imgsz ${esc(r.imgsz || imgsz)}｜耗时 ${elapsed.toFixed(1)}s</span></div>` +
    `<div class="viewer-bar">` +
      `<div class="seg" data-role="shape">` +
        `<button class="seg-btn active" data-shape="box">框</button>` +
        `<button class="seg-btn" data-shape="fill">实心</button>` +
        `<button class="seg-btn" data-shape="dot">中心点</button>` +
      `</div>` +
      `<label>置信度 ≥ <input type="range" data-role="minconf" min="0.01" max="0.99" step="0.01" value="${esc(confUsed)}">` +
      `<b data-role="minconf-val">${esc(confUsed)}</b></label>` +
      `<label><input type="checkbox" data-role="alwaystag"> 常显标签</label>` +
      `<div class="chips" data-role="chips"></div>` +
    `</div>` +
    `<div class="dedup-bar">` +
      `<label><input type="checkbox" data-role="dedupon"${DEDUP_DEFAULT ? ' checked' : ''}> 去重</label>` +
      `<label>中心距 ≤ <input type="number" data-role="dedupdist" value="20" min="0" max="500" step="1"> px</label>` +
      `<label>内嵌率 ≥ <input type="number" data-role="dedupnest" value="80" min="0" max="100" step="5"> %</label>` +
      `<label>重叠时保留 <select data-role="dedupkeep">` +
        `<option value="conf">得分高</option>` +
        `<option value="big">大框（外包）</option>` +
        `<option value="small">小框（内嵌）</option>` +
      `</select></label>` +
    `</div>` +
    `<div class="viewer" data-role="viewer">` +
      `<img class="base" data-role="baseimg" src="${esc(imgUrl)}" alt="推理图">` +
      `<div class="boxes" data-role="boxes"></div>` +
    `</div>` +
    `<div class="viewer-foot">` +
      `<span data-role="count"></span>` +
      (r.result_url ? `<a href="${esc(r.result_url)}" target="_blank">后端渲染图（对照）</a>` : '') +
    `</div>` +
    `<div class="det-table-wrap"><table>` +
      `<thead><tr><th>类别</th><th>置信度</th><th>中心 (cx, cy)</th><th>宽 × 高</th>` +
      `<th>左上右下 (x1, y1, x2, y2)</th></tr></thead>` +
      `<tbody data-role="tablebody"></tbody></table></div>`;

  // 类别胶囊：单击开关，双击只看这一类
  const chips = view.q('[data-role="chips"]');
  if (classes.length) {
    classes.forEach((name) => {
      const b = document.createElement('button');
      b.className = 'chip';
      b.dataset.name = name;
      b.style.background = colorOf.get(name);
      b.textContent = `${name} ${counts.get(name)}`;   // 初值；paintDetects 会按实际显示数刷新
      b.title = '单击：显示 / 隐藏这一类｜双击：只看这一类';
      b.addEventListener('click', () => {
        if (view.hidden.has(name)) view.hidden.delete(name);
        else view.hidden.add(name);
        view.solo = null;
        syncChipState(view);
        paintDetects(view);
      });
      b.addEventListener('dblclick', () => {
        const soloing = view.solo === name;
        view.solo = soloing ? null : name;
        view.hidden = soloing ? new Set() : new Set(classes.filter((n) => n !== name));
        syncChipState(view);
        paintDetects(view);
      });
      chips.appendChild(b);
    });
  } else {
    chips.innerHTML = '<span class="empty-inline">未检出任何目标（可调低推理阈值重跑）</span>';
  }

  // 明细表一次建好，靠 hidden 切换（不重建，避免闪烁）
  const tbody = view.q('[data-role="tablebody"]');
  dets.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.bbox;
    const [cx, cy] = detCenter(d);
    const [w, h] = detSize(d);
    const tr = document.createElement('tr');
    tr.dataset.i = String(i);
    tr.innerHTML =
      `<td>${esc(d.class_name)}</td><td>${esc(d.confidence)}</td>` +
      `<td>${esc(cx)}, ${esc(cy)}</td><td>${esc(w)} × ${esc(h)}</td>` +
      `<td>${esc(x1)}, ${esc(y1)}, ${esc(x2)}, ${esc(y2)}</td>`;
    tr.addEventListener('mouseenter', () => highlightDet(view, i, true));
    tr.addEventListener('mouseleave', () => highlightDet(view, i, false));
    tbody.appendChild(tr);
  });

  // 显示方式：框 / 实心 / 中心点
  view.qs('[data-role="shape"] .seg-btn').forEach((b) => {
    b.addEventListener('click', () => {
      view.shape = b.dataset.shape || 'box';
      view.qs('[data-role="shape"] .seg-btn').forEach((x) => x.classList.toggle('active', x === b));
      paintDetects(view);
    });
  });

  // 二次筛选：不用重跑模型，纯前端过滤
  const slider = view.q('[data-role="minconf"]');
  slider.addEventListener('input', () => {
    view.confMin = Number(slider.value);
    view.q('[data-role="minconf-val"]').textContent = view.confMin.toFixed(2);
    paintDetects(view);
  });
  view.q('[data-role="alwaystag"]').addEventListener('change', (e) => {
    view.q('[data-role="viewer"]').classList.toggle('always-tags', e.target.checked);
  });

  // 去重：纯前端后处理，参数随时可调
  const applyDedup = () => {
    view.dedup = view.q('[data-role="dedupon"]').checked;
    view.dedupDist = Number(view.q('[data-role="dedupdist"]').value) || 0;
    view.dedupNest = (Number(view.q('[data-role="dedupnest"]').value) || 0) / 100;
    view.dedupKeep = view.q('[data-role="dedupkeep"]').value;
    paintDetects(view);
  };
  view.q('[data-role="dedupon"]').addEventListener('change', applyDedup);
  view.q('[data-role="dedupkeep"]').addEventListener('change', applyDedup);
  ['dedupdist', 'dedupnest'].forEach((role) =>
    view.q(`[data-role="${role}"]`).addEventListener('input', applyDedup)
  );

  // 图片就绪后再画框（需要原图宽高把 bbox 换算成百分比）
  const img = view.q('[data-role="baseimg"]');
  const onReady = () => {
    view.naturalW = img.naturalWidth || 1;
    view.naturalH = img.naturalHeight || 1;
    paintDetects(view);
  };
  if (img.complete && img.naturalWidth) onReady();
  else img.addEventListener('load', onReady);
  img.addEventListener('error', () => {
    view.q('[data-role="count"]').textContent = `原图加载失败：${imgUrl}`;
  });
}

/** 让胶囊的亮/灰状态与 view.hidden 保持一致 */
function syncChipState(view) {
  view.qs('[data-role="chips"] .chip').forEach((b) => {
    b.classList.toggle('off', view.hidden.has(b.dataset.name));
  });
}

/** 中心点 (cx, cy)：优先后端 bbox_cxcywh，没有就由 bbox 反算 */
function detCenter(d) {
  if (Array.isArray(d.bbox_cxcywh) && d.bbox_cxcywh.length === 4) {
    return [d.bbox_cxcywh[0], d.bbox_cxcywh[1]];
  }
  const [x1, y1, x2, y2] = d.bbox;
  return [(x1 + x2) / 2, (y1 + y2) / 2];
}

/** 宽高 (w, h)：同上 */
function detSize(d) {
  if (Array.isArray(d.bbox_cxcywh) && d.bbox_cxcywh.length === 4) {
    return [d.bbox_cxcywh[2], d.bbox_cxcywh[3]];
  }
  const [x1, y1, x2, y2] = d.bbox;
  return [x2 - x1, y2 - y1];
}

/** 标签元素（颜色由父节点 color 继承） */
function makeDetTag(d) {
  const tag = document.createElement('span');
  tag.className = 'tag';
  tag.textContent = `${d.class_name} ${Number(d.confidence).toFixed(2)}`;
  return tag;
}

/** 框面积（原图像素） */
function detArea(d) {
  const [x1, y1, x2, y2] = d.bbox;
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

/** 两框中心点的像素距离 */
function centerDist(a, b) {
  const [ax, ay] = detCenter(a);
  const [bx, by] = detCenter(b);
  return Math.hypot(ax - bx, ay - by);
}

/** a 被 b 覆盖的面积占 a 的比例（0~1），用来判断嵌套关系 */
function coveredRatio(a, b) {
  const [ax1, ay1, ax2, ay2] = a.bbox;
  const [bx1, by1, bx2, by2] = b.bbox;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const area = (ax2 - ax1) * (ay2 - ay1);
  return area > 0 ? (iw * ih) / area : 0;
}

/**
 * 去重：只合并「同一类别」的框 —— 中心点太近、或一个框被另一个大半包住时，只留一个。
 * 按 dedupKeep 决定先留谁（得分高 / 大框 / 小框），再贪心吃掉与它冲突的后继。
 * @param {object} view 查看器状态
 * @param {number[]} list 候选下标（已按类别/置信度筛过）
 * 返回保留下来的原始下标（升序）。
 */
function dedupPick(view, list) {
  const rank = {
    conf: (d) => -d.confidence,
    big: (d) => -detArea(d),
    small: (d) => detArea(d),
  }[view.dedupKeep];

  const sorted = [...list].sort((i, j) => rank(view.dets[i]) - rank(view.dets[j]));
  const kept = [];

  for (const i of sorted) {
    const a = view.dets[i];
    let dup = false;
    for (const k of kept) {
      const b = view.dets[k];
      if (a.class_name !== b.class_name) continue;   // 跨类别不合并（人身上可以有手机）
      if (view.dedupDist > 0 && centerDist(a, b) <= view.dedupDist) {
        dup = true;
        break;
      }
      if (
        view.dedupNest > 0 &&
        (coveredRatio(a, b) >= view.dedupNest || coveredRatio(b, a) >= view.dedupNest)
      ) {
        dup = true;
        break;
      }
    }
    if (!dup) kept.push(i);
  }
  return kept.sort((x, y) => x - y);   // 还原原始顺序，保持框的层叠与表格一致
}

/** 给标注节点挂上 hover 高亮 / 点击定位事件 */
function bindDetNode(view, node, i) {
  node.addEventListener('mouseenter', () => highlightDet(view, i, true));
  node.addEventListener('mouseleave', () => highlightDet(view, i, false));
  node.addEventListener('click', () => {
    const tr = view.q(`[data-role="tablebody"] tr[data-i="${i}"]`);
    if (tr) tr.scrollIntoView({ block: 'nearest' });
  });
}

/** 按当前筛选条件重画标注（框 / 实心 / 中心点），并同步明细表 */
function paintDetects(view) {
  if (!view) return;
  const wrap = view.q('[data-role="boxes"]');
  wrap.innerHTML = '';

  const show = [];
  view.dets.forEach((d, i) => {
    if (view.hidden.has(d.class_name)) return;
    if (d.confidence < view.confMin) return;
    show.push(i);
  });

  // 去重是最后一道：先按类别/置信度筛完，再合并重复目标
  const before = show.length;
  const shown = view.dedup ? dedupPick(view, show) : show;
  const merged = before - shown.length;

  shown.forEach((i) => {
    const d = view.dets[i];
    const [x1, y1, x2, y2] = d.bbox;
    // 类别色只写一次：CSS 里边框 / 圆点 / 标签底色都取 var(--c)
    const color = view.colorOf.get(d.class_name) || '#1a73e8';
    const pct = (v, total) => `${(v / total) * 100}%`;

    if (view.shape === 'fill') {
      // 实心：完全不透明填充 + 实线边框（透明度 0，alpha=1），填充色用 --c-soft
      const node = document.createElement('div');
      node.className = 'bbox fill';
      node.dataset.i = String(i);
      node.style.setProperty('--c', color);
      node.style.setProperty(
        '--c-soft',
        classColorA(Math.max(0, view.classes.indexOf(d.class_name)), 1)
      );
      node.style.left = pct(x1, view.naturalW);
      node.style.top = pct(y1, view.naturalH);
      node.style.width = pct(x2 - x1, view.naturalW);
      node.style.height = pct(y2 - y1, view.naturalH);
      node.appendChild(makeDetTag(d));
      bindDetNode(view, node, i);
      wrap.appendChild(node);
    } else if (view.shape === 'dot') {
      // 1) 幽灵框：平时隐藏，hover 这个点时才浮现，用来看清它到底框住多大范围
      const ghost = document.createElement('div');
      ghost.className = 'bbox ghost';
      ghost.dataset.i = String(i);
      ghost.style.setProperty('--c', color);
      ghost.style.left = pct(x1, view.naturalW);
      ghost.style.top = pct(y1, view.naturalH);
      ghost.style.width = pct(x2 - x1, view.naturalW);
      ghost.style.height = pct(y2 - y1, view.naturalH);
      wrap.appendChild(ghost);

      // 2) 点（放在框后面 = 显示在框上面）
      const [cx, cy] = detCenter(d);
      const dot = document.createElement('div');
      dot.className = 'dot';
      dot.dataset.i = String(i);
      dot.style.setProperty('--c', color);
      dot.style.left = pct(cx, view.naturalW);
      dot.style.top = pct(cy, view.naturalH);
      dot.appendChild(document.createElement('i'));
      dot.appendChild(makeDetTag(d));
      bindDetNode(view, dot, i);
      wrap.appendChild(dot);
    } else {
      const node = document.createElement('div');
      node.className = 'bbox';
      node.dataset.i = String(i);
      node.style.setProperty('--c', color);
      node.style.left = pct(x1, view.naturalW);
      node.style.top = pct(y1, view.naturalH);
      node.style.width = pct(x2 - x1, view.naturalW);
      node.style.height = pct(y2 - y1, view.naturalH);
      node.appendChild(makeDetTag(d));
      bindDetNode(view, node, i);
      wrap.appendChild(node);
    }
  });

  const visible = new Set(shown);
  view.q('[data-role="tablebody"]').querySelectorAll('tr').forEach((tr) => {
    tr.hidden = !visible.has(Number(tr.dataset.i));
  });

  // 胶囊上的数字 = 图上实际显示的该类数量，跟下面那句"N 个框"对得上（受筛选/去重影响）
  const shownCounts = new Map();
  shown.forEach((i) => {
    const name = view.dets[i].class_name;
    shownCounts.set(name, (shownCounts.get(name) || 0) + 1);
  });
  view.qs('[data-role="chips"] .chip').forEach((b) => {
    const n = shownCounts.get(b.dataset.name) || 0;
    b.textContent = `${b.dataset.name} ${n}`;
    b.classList.toggle('zero', n === 0);
  });

  const notes = [];
  if (merged > 0) notes.push(`去重合并 ${merged} 个`);
  if (shown.length + merged < view.dets.length) notes.push('部分被类别/置信度筛选隐藏');

  const unit = view.shape === 'dot' ? '个点' : '个框';
  view.q('[data-role="count"]').textContent =
    `图上 ${shown.length} ${unit} / 共检出 ${view.dets.length} 个` +
    (notes.length ? `（${notes.join('；')}）` : '');
}

/** 标注 ↔ 明细表行 双向高亮；中心点模式下顺带浮现它对应的框 */
function highlightDet(view, i, on) {
  view
    .qs(`[data-role="boxes"] [data-i="${i}"]:not(.ghost)`)
    .forEach((n) => n.classList.toggle('hl', on));
  view
    .qs(`[data-role="boxes"] .bbox.ghost[data-i="${i}"]`)
    .forEach((n) => n.classList.toggle('show', on));
  const tr = view.q(`[data-role="tablebody"] tr[data-i="${i}"]`);
  if (tr) tr.classList.toggle('hl', on);
}

/* ==================== 图片推理 ==================== */

async function loadImages() {
  const box = el('imageGrid');
  try {
    const data = await api('/api/images');
    if (!data.images.length) {
      box.innerHTML = '<div class="empty">暂无图片，请先上传</div>';
      return;
    }
    box.innerHTML = '';
    data.images.forEach((img) => {
      const fig = document.createElement('figure');
      fig.title = '点击推理：' + img.filename;
      fig.innerHTML = `<img src="${esc(img.url)}" alt=""><figcaption>${esc(img.filename)}</figcaption>`;

      // 删除按钮（hover 时才露出，点它不会触发推理）
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'del-btn';
      del.textContent = '×';
      del.title = '删除这张图片';
      del.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        if (!confirm(`删除图片「${img.filename}」？`)) return;
        try {
          await api(`/api/images/${encodeURIComponent(img.filename)}`, { method: 'DELETE' }, 15000);
          await loadImages();
        } catch (e) {
          banner(`删除失败：${e instanceof Error ? e.message : String(e)}`);
        }
      });
      fig.appendChild(del);

      // 鼠标悬停时弹出大图（原图比例，不裁剪）
      fig.addEventListener('mouseenter', (ev) => showHoverPreview(img.url, img.filename, ev));
      fig.addEventListener('mousemove', (ev) => {
        hoverEv = ev;
        placeHoverPreview(ev);
      });
      fig.addEventListener('mouseleave', hideHoverPreview);
      fig.addEventListener('click', () => {
        hideHoverPreview();
        detectImage(img.filename);
      });
      box.appendChild(fig);
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

async function detectImage(filename) {
  const box = el('detectResult');
  let ticker = null;
  const t0 = Date.now();

  try {
    // 1) 先取参数（元素缺失会被 catch 住并显示出来）
    const conf = el('detConf').value || '0.38';
    const imgsz = el('detImgsz').value || '640';
    const { weights, isWorld, classes } = readDetectOptions('det');
    const modeText = isWorld
      ? `YOLO-World${classes ? `｜prompt「${classes}」` : '｜prompt=权重内置'}`
      : '普通 YOLO';

    // 切到「推理结果」子页面，并显示当前图片名
    switchSubPane('result');
    el('detResultName').textContent = filename;

    // 2) 实时计时，区分"在跑"和"卡死"
    const paint = () => {
      box.innerHTML =
        `<div>推理中… <b>${((Date.now() - t0) / 1000).toFixed(1)}s</b></div>` +
        `<div style="font-size:12px;color:#80868b;margin-top:6px">` +
        `图片：${esc(filename)}｜置信度 ${esc(conf)}｜imgsz ${esc(imgsz)}｜${esc(modeText)}<br>` +
        `（第一次推理要加载模型；World 切换类别还要编码文字；超过 300 秒自动报超时）</div>`;
    };
    paint();
    ticker = setInterval(paint, 200);

    // 3) 请求后端（图片推理允许长耗时）
    const url =
      `/api/detect/image/${encodeURIComponent(filename)}` +
      `?conf=${encodeURIComponent(conf)}&weights=${encodeURIComponent(weights)}` +
      `&imgsz=${encodeURIComponent(imgsz)}` +
      (classes ? `&classes=${encodeURIComponent(classes)}` : '');
    const r = await api(url, { method: 'POST' }, 300000);

    // 4) 渲染结果（字段缺失时明确报错，而不是静默空白）
    if (!r || !Array.isArray(r.detections)) {
      throw new Error(`后端返回数据不完整：${JSON.stringify(r).slice(0, 200)}`);
    }
    renderDetectResult(r, filename, conf, (Date.now() - t0) / 1000, imgsz);

    // 刷新图库缩略图（失败不影响已显示的结果）
    try {
      await loadImages();
    } catch (e) {
      console.warn('[loadImages] 刷新失败', e);
    }
  } catch (err) {
    showError(box, err, '推理失败');
  } finally {
    // 无论成功失败都停掉计时器，避免"一直推理中"
    if (ticker !== null) clearInterval(ticker);
  }
}

el('refreshImages').addEventListener('click', loadImages);

/* ==================== 视频推理 ==================== */

async function loadVideos() {
  const box = el('videoList');
  try {
    const data = await api('/api/videos');
    if (!data.videos.length) {
      box.innerHTML = '<div class="empty">暂无视频，请先上传</div>';
      return;
    }
    box.innerHTML = '';
    data.videos.forEach((v) => {
      const item = document.createElement('div');
      item.className = 'item';

      const left = document.createElement('span');
      left.className = 'name';
      left.textContent = v.filename;

      const right = document.createElement('span');
      right.className = 'item-actions';

      // 时间轴：展开后选择推理的时间段
      const timeBtn = document.createElement('button');
      timeBtn.className = 'btn small ghost';
      timeBtn.textContent = '时间轴';
      timeBtn.title = '只推理其中一段（如中间 10 分钟）';
      timeBtn.addEventListener('click', () => toggleVideoRange(item, v.filename));

      const btn = document.createElement('button');
      btn.className = 'btn small';
      btn.textContent = '推理';
      btn.addEventListener('click', () => detectVideo(v.filename, btn));

      // 删除按钮
      const del = document.createElement('button');
      del.className = 'btn small danger';
      del.textContent = '删除';
      del.title = '删除这个视频';
      del.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        if (!confirm(`删除视频「${v.filename}」？`)) return;
        try {
          await api(`/api/videos/${encodeURIComponent(v.filename)}`, { method: 'DELETE' }, 15000);
          await loadVideos();
        } catch (e) {
          banner(`删除失败：${e instanceof Error ? e.message : String(e)}`);
        }
      });

      right.append(timeBtn, btn, del);
      item.appendChild(left);
      item.appendChild(right);
      box.appendChild(item);
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

/* 视频时间轴：展开/收起「双滑块」设置区。
 * 滑块下限固定 0、上限 = 视频总时长（读取失败时按 1 小时兜底），
 * 拖动两个手柄可选中任意中间区间。
 */
async function toggleVideoRange(item, filename) {
  const existing = item.querySelector('.range-box');
  if (existing) {
    existing.remove();
    return;
  }
  const boxEl = document.createElement('div');
  boxEl.className = 'range-box';
  boxEl.innerHTML =
    `<span class="r-label">0:00</span>` +
    `<div class="dual-range">` +
      `<input type="range" class="r-start" min="0" max="3600" value="0" step="1">` +
      `<input type="range" class="r-end" min="0" max="3600" value="3600" step="1">` +
    `</div>` +
    `<span class="r-label">1:00:00</span>` +
    `<span class="r-hint">读取时长中…</span>`;
  item.appendChild(boxEl);

  const startInput = boxEl.querySelector('.r-start');
  const endInput = boxEl.querySelector('.r-end');
  const labels = boxEl.querySelectorAll('.r-label');
  const hint = boxEl.querySelector('.r-hint');
  const track = boxEl.querySelector('.dual-range');
  let dur = null;   // 视频总时长（秒）

  // 高亮选中区间 + 刷新两端的 mm:ss 显示
  const paint = () => {
    const max = dur || Number(startInput.max);
    const pct = (v) => (max > 0 ? (v / max) * 100 : 0);
    const s = Number(startInput.value);
    const e = Number(endInput.value);
    labels[0].textContent = fmtDuration(s);
    labels[1].textContent = fmtDuration(e);
    track.style.background =
      `linear-gradient(to right, #e8eaed ${pct(s)}%, #1a73e8 ${pct(s)}%, ` +
      `#1a73e8 ${pct(e)}%, #e8eaed ${pct(e)}%)`;
  };

  // 保证 start <= end：谁在动就把它钳到另一个的值
  const clampPair = () => {
    let s = Number(startInput.value);
    let e = Number(endInput.value);
    if (s > e) {
      if (document.activeElement === startInput) s = e;
      else e = s;
      startInput.value = String(s);
      endInput.value = String(e);
    }
    paint();
  };
  startInput.addEventListener('input', clampPair);
  endInput.addEventListener('input', clampPair);

  // 拉取元信息：把滑块上限设成真实总时长
  try {
    const info = await api(`/api/videos/${encodeURIComponent(filename)}/info`, {}, 10000);
    if (info && info.duration > 0) {
      dur = Math.floor(info.duration);
      startInput.max = String(dur);
      endInput.max = String(dur);
      endInput.value = String(dur);
      hint.textContent = `总长 ${fmtDuration(dur)}，拖动两端选择区间`;
    } else {
      hint.textContent = '无法读取时长，按 1 小时上限选择';
    }
  } catch (e) {
    console.warn('[视频元信息] 获取失败', e);
    hint.textContent = '无法读取时长，按 1 小时上限选择';
  }
  paint();
}

/** 秒 → mm:ss（超过 1 小时显示 h:mm:ss） */
function fmtDuration(sec) {
  sec = Math.max(0, Math.floor(Number(sec) || 0));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(s).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

/** 读取该视频项当前设置的时间段（从展开的双滑块里找；没展开就用全片） */
function readVideoRange(item) {
  const r = item.querySelector('.range-box');
  if (!r) return { start: 0, end: null };
  const s = Number(r.querySelector('.r-start').value) || 0;
  const e = Number(r.querySelector('.r-end').value);
  const max = Number(r.querySelector('.r-end').max);
  // start=0 且 end 拖到最右 = 全片
  if (s === 0 && e >= max) return { start: 0, end: null };
  return { start: s, end: e >= max ? max : e };
}

async function detectVideo(filename, btn) {
  const box = el('videoResult');
  let ticker = null;
  const t0 = Date.now();

  // 找到对应的列表项（推理按钮的父级 .item），读它自己的时间段
  const item = btn.closest('.item');

  try {
    const conf = el('vidConf').value || '0.38';
    const stride = el('vidStride').value || '1';
    const render = el('vidRender').value || 'box';
    const { start, end } = readVideoRange(item);
    const { weights, isWorld, classes } = readDetectOptions('vid');
    const rangeText = end == null ? `全片` : `${fmtDuration(start)} ~ ${fmtDuration(end)}`;
    const renderText = render === 'dot' ? '中心点' : '框';
    const modeText = isWorld
      ? `YOLO-World${classes ? `｜prompt「${classes}」` : '｜prompt=权重内置'}`
      : '普通 YOLO';

    btn.disabled = true;
    box.innerHTML =
      `<div>提交中… <b>${((Date.now() - t0) / 1000).toFixed(1)}s</b></div>` +
      `<div style="font-size:12px;color:#80868b">视频：${esc(filename)}｜时间段 ${rangeText}｜显示 ${renderText}｜置信度 ${esc(conf)}｜抽帧间隔 ${esc(stride)}｜${esc(modeText)}</div>`;

    // 提交任务：30 秒内必须返回 job_id
    const q = new URLSearchParams({
      conf,
      stride,
      weights,
      start: String(start),
      render,
    });
    if (end != null) q.set('end', String(end));
    if (classes) q.set('classes', classes);
    const res = await api(
      `/api/detect/video/${encodeURIComponent(filename)}?${q}`,
      { method: 'POST' },
      30000
    );
    if (!res || !res.job_id) {
      throw new Error(`后端未返回 job_id：${JSON.stringify(res).slice(0, 200)}`);
    }

    // 轮询期间保持计时
    ticker = setInterval(() => {
      const node = box.querySelector('.elapsed');
      if (node) node.textContent = `${((Date.now() - t0) / 1000).toFixed(1)}s`;
    }, 200);

    pollJob(
      res.job_id,
      box,
      (job) => {
        btn.disabled = false;
        if (job.status === 'cancelled') {
          box.innerHTML =
            `<div style="color:#f9a825;font-weight:600">已取消推理任务。</div>`;
        } else if (job.result_url) {
          box.innerHTML = `<video src="${esc(job.result_url)}" controls autoplay></video>`;
          // 折线图单独面板显示（不塞在视频下面，避免被大视频挤到看不见）
          showHeadChart(job.chart_url, filename);
        } else {
          box.innerHTML =
            `<div style="color:#c5221f;font-weight:600">视频推理未成功：${esc(job.message)}</div>`;
        }
        loadVideos();
        loadOutputs();
      },
      t0
    );
  } catch (err) {
    btn.disabled = false;
    showError(box, err, '视频推理失败');
  } finally {
    if (ticker !== null) clearInterval(ticker);
  }
}

el('refreshVideos').addEventListener('click', loadVideos);

/* ==================== 推理结果库（data/outputs/）==================== */

/**
 * 在独立面板里显示人数折线图（视频页）
 * @param {string} chartUrl 折线图数据 json（无则显示提示）
 * @param {string} title    面板标题后缀（一般传结果视频名）
 */
function showHeadChart(chartUrl, title = '') {
  const panel = document.getElementById('chartPanel');
  const host = document.getElementById('chartHost');
  // 兜底：页面是旧缓存（没有面板）时，退回到塞在推理结果区里，保证能看到图
  if (!panel || !host) {
    const fallback = el('videoResult');
    if (chartUrl) renderHeadChart(fallback, chartUrl);
    return;
  }
  const titleNode = document.getElementById('chartTitle');
  if (titleNode) titleNode.textContent = title ? `— ${title}` : '';
  panel.hidden = false;
  host.innerHTML = '';
  if (!chartUrl) {
    host.innerHTML =
      '<div class="headchart-empty">该结果没有折线图数据（旧结果或检测时未记录）</div>';
  } else {
    renderHeadChart(host, chartUrl);
  }
  // 面板在页面下方，滚进视野，否则用户以为"没出现"
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/** 收起折线图面板 */
function hideHeadChart() {
  const panel = document.getElementById('chartPanel');
  const host = document.getElementById('chartHost');
  if (panel) panel.hidden = true;
  if (host) host.innerHTML = '';
}

const _chartCloseBtn = document.getElementById('chartClose');
if (_chartCloseBtn) _chartCloseBtn.addEventListener('click', hideHeadChart);

/**
 * 人数折线图（每帧 head 检出数）—— 像剪辑时间轴一样：
 *  - 横向滚动（外层 overflow-x: auto），鼠标按住拖动可平移
 *  - 按住 Ctrl + 滚轮：缩放「每帧占的像素宽」，以鼠标位置为锚点（视口不漂移）
 *  - X 轴显示时间（按 fps 由帧号换算），悬停显示该帧人数
 * @param {Element} host 渲染到哪个容器
 * @param {string} chartUrl 折线图数据 json（{head_counts:[[frame,count]], fps, start_frame}）
 */
function renderHeadChart(host, chartUrl) {
  host.innerHTML = '';

  const wrap = document.createElement('div');
  wrap.className = 'headchart';
  wrap.innerHTML =
    `<div class="headchart-head">` +
    `<span class="headchart-hint">⇔ 滚动/拖动平移 ｜ Ctrl+滚轮 缩放宽度</span></div>` +
    `<div class="headchart-scroll"><div class="headchart-track"></div></div>`;
  host.appendChild(wrap);

  const scroll = wrap.querySelector('.headchart-scroll');
  const track = wrap.querySelector('.headchart-track');

  // 几何常量
  const PAD_L = 40, PAD_R = 14, PAD_T = 12, PAD_B = 24, H = 150;
  let pxPerFrame = 6;          // 每帧占的像素宽（Ctrl+滚轮调整）
  let scale = 6;               // 实际像素/帧（draw 里由 pxPerFrame + 最小宽度约束求得）
  let view = null;             // 当前视图数据（draw 时填充）

  /** 帧号 → 该帧在 track 里的像素 x（依赖当前 scale） */
  function frameToX(frame) {
    if (!view) return PAD_L;
    return PAD_L + (frame - view.firstFrame) * scale;
  }

  /** track 像素 x → 帧号（frameToX 的逆） */
  function xToFrame(px) {
    if (!view || !scale) return 0;
    return view.firstFrame + (px - PAD_L) / scale;
  }

  /** 帧号 → 时间文本（秒，超过 1 小时显示 h:mm:ss） */
  function fmtFrameTime(frame) {
    const fps = view && view.fps ? view.fps : 25;
    const sec = Math.max(0, frame / fps);
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    const mm = String(m).padStart(2, '0');
    const ss = String(s).padStart(2, '0');
    return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
  }

  // 滚轮：Ctrl → 缩放宽度；不带 Ctrl → 纵向滚轮转成横向平移（时间轴习惯）
  scroll.addEventListener(
    'wheel',
    (e) => {
      if (!e.ctrlKey) {
        // 已滚到两端时不拦截，让页面正常滚动，避免"卡住"感
        const maxScroll = scroll.scrollWidth - scroll.clientWidth;
        if (e.deltaY === 0 || maxScroll <= 0) return;
        if (
          (e.deltaY > 0 && scroll.scrollLeft >= maxScroll - 1) ||
          (e.deltaY < 0 && scroll.scrollLeft <= 0)
        ) {
          return;
        }
        e.preventDefault();
        scroll.scrollLeft += e.deltaY;
        return;
      }
      e.preventDefault();
      if (!view) return;
      const rect = scroll.getBoundingClientRect();
      const pointerX = e.clientX - rect.left;      // 视口内坐标
      const posInTrack = pointerX + scroll.scrollLeft;
      const anchorFrame = xToFrame(posInTrack);    // 鼠标下的帧
      const factor = e.deltaY > 0 ? 0.85 : 1.18;
      pxPerFrame = Math.min(60, Math.max(0.5, pxPerFrame * factor));
      draw();
      // 缩放后把锚点帧滚回鼠标原来的位置（精确补偿，视口不漂移）
      scroll.scrollLeft = frameToX(anchorFrame) - pointerX;
    },
    { passive: false }
  );

  // 按住拖动平移（像时间轴一样）
  let dragging = false, dragStartX = 0, dragStartScroll = 0;
  scroll.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    dragging = true;
    dragStartX = e.clientX;
    dragStartScroll = scroll.scrollLeft;
    scroll.classList.add('dragging');
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    scroll.scrollLeft = dragStartScroll - (e.clientX - dragStartX);
  });
  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    scroll.classList.remove('dragging');
  });

  function draw() {
    const points = view ? view.points : [];
    if (!points.length) {
      track.innerHTML = '<div class="headchart-empty">无数据</div>';
      track.style.width = '100%';
      return;
    }
    view.firstFrame = points[0][0];
    view.lastFrame = points[points.length - 1][0];
    const range = Math.max(1, view.lastFrame - view.firstFrame);
    const maxCount = Math.max(1, ...points.map((p) => p[1]));

    const plotW = Math.max(200, range * pxPerFrame);
    scale = plotW / range;      // 统一比例：保证折线始终铺满绘图区
    const width = PAD_L + PAD_R + plotW;
    const plotH = H - PAD_T - PAD_B;
    const y = (cnt) => PAD_T + plotH - (cnt / maxCount) * plotH;

    // 折线
    const poly = points
      .map(([fr, cnt]) => `${frameToX(fr).toFixed(1)},${y(cnt).toFixed(1)}`)
      .join(' ');

    // 横向网格 + Y 轴刻度（0 / 一半 / 最大）
    let grid = '';
    [0, 0.5, 1].forEach((ratio) => {
      const cnt = Math.round(maxCount * ratio);
      const yy = y(cnt).toFixed(1);
      grid += `<line x1="${PAD_L}" y1="${yy}" x2="${width - PAD_R}" y2="${yy}" class="hg-grid"/>`;
      grid += `<text x="${PAD_L - 6}" y="${Number(yy) + 4}" class="hg-y">${cnt}</text>`;
    });

    // X 轴时间刻度：按「好看的」时间步长取，目标 6~10 个刻度
    const fps = view.fps || 25;
    const totalSec = range / fps;
    const niceSteps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800];
    const stepSec = niceSteps.find((s) => totalSec / s <= 9) || Math.ceil(totalSec / 8);
    let xticks = '';
    const startSec = view.firstFrame / fps;
    const firstTick = Math.ceil(startSec / stepSec) * stepSec;   // 对齐到整数刻度
    for (let sec = firstTick; sec <= startSec + totalSec + 1e-6; sec += stepSec) {
      const fr = sec * fps;
      const xx = frameToX(fr).toFixed(1);
      const label = sec >= 3600
        ? `${Math.floor(sec / 3600)}:${String(Math.floor((sec % 3600) / 60)).padStart(2, '0')}:${String(Math.floor(sec % 60)).padStart(2, '0')}`
        : `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, '0')}`;
      xticks += `<line x1="${xx}" y1="${PAD_T}" x2="${xx}" y2="${H - PAD_B}" class="hg-grid"/>`;
      xticks += `<text x="${xx}" y="${H - 7}" class="hg-x">${label}</text>`;
    }

    track.innerHTML =
      `<svg width="${width}" height="${H}" class="hg-svg">` +
      grid + xticks +
      `<polyline points="${poly}" class="hg-line" fill="none"/>` +
      `<line class="hg-cursor" x1="0" y1="${PAD_T}" x2="0" y2="${H - PAD_B}" style="display:none"/>` +
      `<circle class="hg-dot" r="3.5" style="display:none"/>` +
      `<g class="hg-tip" style="display:none">` +
        `<rect class="hg-tip-bg" rx="3" width="0" height="16"/>` +
        `<text class="hg-tip-tx" y="11"></text>` +
      `</g>` +
      `</svg>`;
    track.style.width = width + 'px';
  }

  /**
   * 悬停：竖线 + 最近数据点 + 人数气泡。
   * 只绑定一次（每次 draw 会重建 SVG 元素，所以这里按需 query，不缓存节点）。
   */
  scroll.addEventListener('mousemove', (e) => {
    if (dragging || !view || !view.points.length) return;
    const svg = track.querySelector('svg.hg-svg');
    if (!svg) return;
    const cursor = svg.querySelector('.hg-cursor');
    const dot = svg.querySelector('.hg-dot');
    const tip = svg.querySelector('.hg-tip');
    const tipBg = svg.querySelector('.hg-tip-bg');
    const tipTx = svg.querySelector('.hg-tip-tx');
    if (!cursor || !dot || !tip || !tipBg || !tipTx) return;

    const rect = scroll.getBoundingClientRect();
    const posInTrack = e.clientX - rect.left + scroll.scrollLeft;
    // 线性搜索最近点（点数不多，够用）
    let nearest = view.points[0];
    let bestD = Infinity;
    view.points.forEach((p) => {
      const d = Math.abs(frameToX(p[0]) - posInTrack);
      if (d < bestD) { bestD = d; nearest = p; }
    });

    const maxCount = Math.max(1, ...view.points.map((p) => p[1]));
    const plotH = H - PAD_T - PAD_B;
    const xx = frameToX(nearest[0]);
    const yy = PAD_T + plotH - (nearest[1] / maxCount) * plotH;

    cursor.setAttribute('x1', xx);
    cursor.setAttribute('x2', xx);
    cursor.style.display = '';
    dot.setAttribute('cx', xx);
    dot.setAttribute('cy', yy);
    dot.style.display = '';

    const label = `${fmtFrameTime(nearest[0])} · ${nearest[1]} 人`;
    tipTx.textContent = label;
    const tw = label.length * 6.5 + 10;   // 粗略估宽
    tipBg.setAttribute('width', tw);
    const flip = xx + tw + 8 > Number(svg.getAttribute('width')) - PAD_R;
    const tx = flip ? xx - tw - 6 : xx + 6;
    tip.setAttribute('transform', `translate(${tx},${Math.max(2, yy - 22)})`);
    tip.style.display = '';
  });
  scroll.addEventListener('mouseleave', () => {
    const svg = track.querySelector('svg.hg-svg');
    if (!svg) return;
    ['.hg-cursor', '.hg-dot', '.hg-tip'].forEach((sel) => {
      const n = svg.querySelector(sel);
      if (n) n.style.display = 'none';
    });
  });

  api(chartUrl, {}, 15000)
    .then((d) => {
      const raw = (d && Array.isArray(d.head_counts)) ? d.head_counts : [];
      view = {
        all: raw,
        points: condense(raw, 5000),   // 点太多时降采样，避免 SVG 卡顿
        fps: (d && d.fps) || 25,
        firstFrame: raw.length ? raw[0][0] : 0,
        lastFrame: raw.length ? raw[raw.length - 1][0] : 0,
      };
      // 标题补充总帧数
      const hint = wrap.querySelector('.headchart-hint');
      if (hint && raw.length) {
        hint.textContent = `共 ${raw.length} 个采样帧 ｜ ⇔ 滚动/拖动平移 ｜ Ctrl+滚轮 缩放`;
      }
      // 初始宽度：让整条曲线大约铺满视口（留一点余量，可继续缩放）
      const range = Math.max(1, view.lastFrame - view.firstFrame);
      const viewportW = scroll.clientWidth || 600;
      pxPerFrame = Math.min(12, Math.max(0.5, (viewportW - PAD_L - PAD_R - 40) / range));
      draw();
    })
    .catch((e) => {
      track.innerHTML = `<div class="headchart-empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
    });
}

/**
 * 降采样：点数超过 maxPoints 时按桶聚合，每桶保留「人数最多」的点（保住峰值形状）。
 * 用于长时间视频（逐帧记录可能上万点），避免 SVG 渲染卡顿。
 */
function condense(points, maxPoints) {
  if (points.length <= maxPoints) return points;
  const bucket = Math.ceil(points.length / maxPoints);
  const out = [];
  for (let i = 0; i < points.length; i += bucket) {
    let best = points[i];
    const end = Math.min(i + bucket, points.length);
    for (let j = i + 1; j < end; j++) {
      if (points[j][1] > best[1]) best = points[j];
    }
    out.push(best);
  }
  return out;
}

async function loadOutputs() {
  const box = el('outputList');
  try {
    const data = await api('/api/videos');
    const outs = data.outputs || [];
    if (!outs.length) {
      box.innerHTML = '<div class="empty">暂无推理结果视频</div>';
      return;
    }
    box.innerHTML = '';
    outs.forEach((o) => {
      const item = document.createElement('div');
      item.className = 'item';
      item.innerHTML = `<span class="name">${esc(o.filename)}</span>`;

      const right = document.createElement('span');
      right.className = 'item-actions';

      // 预览：内嵌 <video>；折线图在独立面板里显示
      const play = document.createElement('button');
      play.className = 'btn small ghost';
      play.textContent = '预览';
      play.addEventListener('click', () => {
        const r = el('videoResult');
        r.innerHTML = `<video src="${esc(o.url)}" controls autoplay></video>`;
        showHeadChart(o.chart_url, o.filename);
      });

      // 图预览：该段「召回率最高帧」已提取到待推理图片，点它切过去并自动推理
      let pv = null;
      if (o.best_frame) {
        pv = document.createElement('button');
        pv.className = 'btn small ghost';
        pv.textContent = '图预览';
        pv.title = '把该段检出目标数最多的一帧提取到「待推理图片」，并自动推理';
        pv.addEventListener('click', async () => {
          // 切到图片推理 tab 的「待推理图片」子页
          document.querySelector('.tab[data-tab="detect"]').click();
          switchSubPane('images');
          await loadImages();
          await detectImage(o.best_frame);
        });
        right.appendChild(pv);
      }

      // 删除
      const del = document.createElement('button');
      del.className = 'btn small danger';
      del.textContent = '删除';
      del.addEventListener('click', async () => {
        if (!confirm(`删除推理结果「${o.filename}」？`)) return;
        try {
          await api(`/api/outputs/${encodeURIComponent(o.filename)}`, { method: 'DELETE' }, 15000);
          await loadOutputs();
        } catch (e) {
          banner(`删除失败：${e instanceof Error ? e.message : String(e)}`);
        }
      });

      right.append(play, del);
      item.appendChild(right);
      box.appendChild(item);
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

el('refreshOutputs').addEventListener('click', loadOutputs);
loadOutputs();

/* ==================== 训练 ==================== */

async function loadDatasets() {
  const box = el('datasetList');
  const sel = el('trainDataset');
  const annSel = el('annotateDataset');
  try {
    const data = await api('/api/datasets');
    sel.innerHTML = '';
    annSel.innerHTML = '';
    if (!data.datasets.length) {
      box.innerHTML = '<div class="empty">暂无数据集</div>';
      sel.innerHTML = '<option value="">（无数据集）</option>';
      annSel.innerHTML = '<option value="">（无数据集）</option>';
      return;
    }
    box.innerHTML = '';
    data.datasets.forEach((d) => {
      const item = document.createElement('div');
      item.className = 'item';
      item.innerHTML =
        `<span class="name">${esc(d.name)}</span>` +
        `<span class="meta">${esc(d.images)} 张图片${d.has_yaml ? ' · 已有 yaml' : ''}</span>`;
      box.appendChild(item);

      for (const s of [sel, annSel]) {
        const o = document.createElement('option');
        o.value = d.name;
        o.textContent = d.name;
        s.appendChild(o);
      }
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

el('trainBtn').addEventListener('click', async () => {
  const box = el('trainResult');
  const dataset = el('trainDataset').value;

  if (!dataset) {
    box.innerHTML = '<span style="color:#c5221f">请先准备数据集（data/datasets/&lt;名称&gt;/）</span>';
    return;
  }

  const params = new URLSearchParams({
    dataset,
    weights: el('trainWeights').value || DEFAULT_WEIGHTS,
    epochs: el('trainEpochs').value,
    imgsz: el('trainImgsz').value,
    batch: el('trainBatch').value,
    lr0: el('trainLr0').value,
  });

  box.innerHTML = '正在启动训练…';
  const t0 = Date.now();

  try {
    const res = await api(`/api/train?${params}`, { method: 'POST' }, 30000);
    if (!res || !res.job_id) {
      throw new Error(`后端未返回 job_id：${JSON.stringify(res).slice(0, 200)}`);
    }

    pollJob(
      res.job_id,
      box,
      (job) => {
        let html = `<div class="summary">${esc(job.message)}</div>`;
        if (job.metrics && Object.keys(job.metrics).length) {
          html += '<table><tr><th>指标</th><th>值</th></tr>';
          Object.entries(job.metrics).forEach(([k, v]) => {
            html += `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`;
          });
          html += '</table>';
        }
        if (job.result_url) {
          html +=
            `<p style="margin-top:10px">最佳权重：` +
            `<a href="${esc(job.result_url)}" download>下载 best.pt</a></p>`;
        }
        box.innerHTML = html;
        loadEnv();
      },
      t0
    );
  } catch (err) {
    showError(box, err, '训练启动失败');
  }
});

/* ==================== 标注（全屏标注器）====================
 * 入口：训练页「标注数据」区选数据集 + Train/Val → 点缩略图进入全屏标注器。
 * 能力：预标注（用与推理一致的权重选项）→ 框/点两种显示 → 改框（拖动/缩放）/
 *       改类别 / 删框 / 画新框 → 保存为 YOLO label。
 */

let ANN = null;   // 当前标注会话

/** 类别下拉里「重命名本类」那个特殊选项的值（不是真的 class_id） */
const RENAME_OPT = '__rename__';

/** 缩略图网格：列出数据集某子集的图片 + 已标注状态 */
async function loadAnnotateGrid() {
  const grid = el('annotateGrid');
  const stat = el('annotateStat');
  const ds = el('annotateDataset').value;
  const split = el('annotateSplit').value;

  if (!ds) {
    grid.innerHTML = '<div class="empty">暂无数据集</div>';
    stat.textContent = '';
    return;
  }

  grid.innerHTML = '<div class="empty">加载中…</div>';
  try {
    const data = await api(`/api/datasets/${encodeURIComponent(ds)}`, {}, 20000);
    const sub = (data.subsets && data.subsets[split]) || null;
    if (!sub || !sub.count) {
      grid.innerHTML = `<div class="empty">该子集没有图片（${esc(ds)}/images/${esc(split)}）</div>`;
      stat.textContent = '';
      return;
    }
    stat.textContent = `${sub.count} 张 · 已标注 ${sub.labeled} 张 · ${(data.classes || []).length} 类`;

    // 已标注状态按需读取（点开图片后才知），这里只列缩略图
    grid.innerHTML = '';
    sub.files.forEach((f) => {
      const fig = document.createElement('figure');
      fig.title = '点击进入全屏标注：' + f;
      fig.innerHTML =
        `<img src="/api/annotate/raw/${encodeURIComponent(ds)}/${encodeURIComponent(split)}/${encodeURIComponent(f)}" alt="" loading="lazy" draggable="false">` +
        `<figcaption>${esc(f)}</figcaption>`;
      fig.addEventListener('click', () => openAnnotator(ds, split, f));
      grid.appendChild(fig);
    });
  } catch (e) {
    grid.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

el('annotateDataset').addEventListener('change', loadAnnotateGrid);
el('annotateSplit').addEventListener('change', loadAnnotateGrid);
el('refreshAnnotate').addEventListener('click', loadAnnotateGrid);

/** 打开全屏标注器，载入该图片的现有标注 */
async function openAnnotator(ds, split, filename) {
  const mask = el('annotatorMask');
  const rel = `${ds}/${split}/${filename}`;

  ANN = {
    rel,
    dsName: ds,        // 数据集名（改类别名时要写进 data.yaml）
    split,
    dsClasses: [],
    dets: [],          // [{class_id, class_name, confidence, bbox:[x1,y1,x2,y2]}]
    active: -1,        // 当前选中框
    shape: 'box',      // 'box' | 'dot'
    showLabels: el('annShowLabels').checked,   // 标签开关（跨图片记住上次选择）
    naturalW: 1,
    naturalH: 1,
    filter: null,      // 只看某一类（null = 全部）
    confMin: 0,
    dirty: false,
  };

  mask.hidden = false;
  el('annotatorTitle').textContent = filename;
  el('annotatorSub').textContent = `${ds} / ${split}`;
  el('annList').innerHTML = '';
  el('annClassesList').innerHTML = '';
  el('annCount').textContent = '';
  annMsg('载入中…');

  // 先把图片地址写上（立即开始加载）
  const img = el('annBaseImg');
  img.src = `/api/annotate/raw/${encodeURIComponent(ds)}/${encodeURIComponent(split)}/${encodeURIComponent(filename)}`;

  try {
    const data = await api(
      `/api/annotate/image/${encodeURIComponent(ds)}/${encodeURIComponent(split)}/${encodeURIComponent(filename)}`,
      {},
      60000
    );
    ANN.dsClasses = data.classes || [];
    ANN.dets = (data.detections || []).map((d) => ({
      class_id: d.class_id,
      class_name: d.class_name,
      confidence: d.confidence,
      bbox: d.bbox.slice(),
    }));
    ANN.naturalW = data.width || 1;
    ANN.naturalH = data.height || 1;

    // 图片加载完后按真实尺寸画框（百分比定位）
    const onReady = () => {
      ANN.naturalW = img.naturalWidth || ANN.naturalW;
      ANN.naturalH = img.naturalHeight || ANN.naturalH;
      paintAnn();
    };
    if (img.complete && img.naturalWidth) onReady();
    else img.addEventListener('load', onReady, { once: true });

    paintAnnClasses();
    paintAnn();
    annMsg('');
  } catch (e) {
    annMsg(`载入失败：${e instanceof Error ? e.message : String(e)}`);
  }
}

/** 顶部提示条 */
function annMsg(text) {
  el('annMsg').textContent = text || '';
}

/** 左侧类别筛选列表（每个类别带一个 ✎ 改名按钮 + 底部「新增类别」） */
function paintAnnClasses() {
  const box = el('annClassesList');
  box.innerHTML = '';
  ANN.dsClasses.forEach((name, i) => {
    const row = document.createElement('span');
    row.className = 'ann-class-row';

    const b = document.createElement('button');
    b.className = 'ann-class';
    b.style.setProperty('--c', classColor(i));
    b.textContent = name;
    b.title = '单击：只看这一类（再次点击取消）';
    b.classList.toggle('active', ANN.filter === i);
    b.addEventListener('click', () => {
      ANN.filter = ANN.filter === i ? null : i;
      paintAnnClasses();
      paintAnn();
    });

    const ed = document.createElement('button');
    ed.className = 'ann-class-edit';
    ed.type = 'button';
    ed.textContent = '✎';
    ed.title = `重命名类别「${name}」`;
    ed.addEventListener('click', (ev) => {
      ev.stopPropagation();
      void renameClass(i);
    });

    row.appendChild(b);
    row.appendChild(ed);
    box.appendChild(row);
  });

  // 新增类别：类别的增删改都只能在这里做（不再有 classes.txt 可以手改），
  // 改完由后端直接写进 data.yaml 的 names + nc
  const add = document.createElement('button');
  add.className = 'ann-class-add';
  add.type = 'button';
  add.textContent = '＋ 新增类别';
  add.title = '新增一个类别（会写入 data.yaml）';
  add.addEventListener('click', (ev) => {
    ev.stopPropagation();
    void addClass();
  });
  box.appendChild(add);
}

/** 新增一个类别：追加到末尾 → class_id 顺延，不影响已有 label */
async function addClass() {
  if (!ANN) return;
  const input = prompt('新增类别名：', '');
  if (input === null) return;
  const name = input.trim();
  if (!name) return;
  if (ANN.dsClasses.includes(name)) {
    annMsg(`类别「${name}」已存在`);
    return;
  }
  await saveClasses([...ANN.dsClasses, name], `已新增类别「${name}」（id=${ANN.dsClasses.length}）`);
}

/** 重命名数据集的第 i 个类别
 *
 * 只改名字：class_id（data.yaml 里 names 的下标）不变，所以已有 label 文件
 * 里的数字不需要重写。后端会一并写 data.yaml 的 names/nc 并清训练标注缓存。
 */
async function renameClass(i) {
  if (!ANN || i < 0 || i >= ANN.dsClasses.length) return;
  const old = ANN.dsClasses[i];
  const input = prompt(`重命名类别「${old}」为：`, old);
  if (input === null) return;               // 取消
  const name = input.trim();
  if (!name || name === old) return;
  if (ANN.dsClasses.includes(name)) {
    annMsg(`类别「${name}」已存在，不能重名`);
    return;
  }
  const names = ANN.dsClasses.slice();
  names[i] = name;
  await saveClasses(names, `已改名为「${name}」（class_id 不变，无需重新保存标注）`);
}

/** 把整份类别清单提交给后端（改名 / 新增都走这里） */
async function saveClasses(names, okMsg) {
  if (!ANN) return;
  annMsg('正在更新类别…');
  try {
    const res = await api(
      `/api/datasets/${encodeURIComponent(ANN.dsName)}/classes`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ classes: names }),
      },
      30000
    );
    if (!ANN) return;   // 提交期间用户可能已关掉标注器
    ANN.dsClasses = res.classes || names;
    // 已载入的标注跟着换显示名（class_id 未变，label 文件无需重写）
    ANN.dets.forEach((d) => {
      d.class_name = ANN.dsClasses[d.class_id] || String(d.class_id);
    });
    paintAnnClasses();
    paintAnn();
    annMsg(okMsg);
    loadAnnotateGrid();
  } catch (e) {
    if (ANN) annMsg(`更新类别失败：${e instanceof Error ? e.message : String(e)}`);
  }
}

/** 类别 → 调色用的下标（与推理结果页同一套黄金角取色） */
function annClassIndex(clsId, name) {
  let idx = clsId;
  if (typeof idx !== 'number' || idx < 0) {
    idx = ANN.dsClasses.indexOf(name);
  }
  return idx < 0 ? 0 : idx;
}

/** 重画所有标注框（框/点两种形式）+ 左侧列表 */
function paintAnn() {
  if (!ANN) return;
  const wrap = el('annBoxes');
  wrap.innerHTML = '';

  // 标签开关：关掉后任何模式下都不显示类别标签
  // （点模式 hover 时也只出现框，不出现标签）
  wrap.classList.toggle('no-labels', !ANN.showLabels);

  const visible = [];

  ANN.dets.forEach((d, i) => {
    if (ANN.filter !== null && d.class_id !== ANN.filter) return;
    visible.push(i);

    const ci = annClassIndex(d.class_id, d.class_name);
    const color = classColor(ci);
    const node = document.createElement('div');
    node.className = 'ann-box' + (ANN.active === i ? ' active' : '');
    node.dataset.i = String(i);
    node.style.setProperty('--c', color);
    // 点模式下框要半透明，但中心点必须不透明。
    // 不能用父节点的 opacity —— 那会把里面的中心点一起变淡，
    // 所以单独给边框一个带 alpha 的颜色变量（--c-soft）。
    node.style.setProperty('--c-soft', classColorA(ci, 0.55));
    // 拖拽/新建过程中不要整层重建 DOM，否则手感发顿（不跟手）
    applyAnnBoxRect(node, d.bbox);

    // 类别标签：下拉直接切换；最后一项是「重命名本类…」
    const sel = document.createElement('select');
    sel.className = 'ann-cls-select';
    sel.title = '切换类别，或选最后一项重命名这个类别';
    ANN.dsClasses.forEach((name, ci) => {
      const o = document.createElement('option');
      o.value = String(ci);
      o.textContent = name;
      if (ci === d.class_id) o.selected = true;
      sel.appendChild(o);
    });
    const rn = document.createElement('option');
    rn.value = RENAME_OPT;
    rn.textContent = '✎ 重命名本类…';
    sel.appendChild(rn);
    sel.addEventListener('change', (ev) => {
      ev.stopPropagation();
      if (sel.value === RENAME_OPT) {
        void renameClass(d.class_id);
        return;               // renameClass 里会重画，不用在这里改 class_id
      }
      d.class_id = Number(sel.value);
      d.class_name = ANN.dsClasses[d.class_id] || String(d.class_id);
      ANN.dirty = true;
      paintAnn();
    });
    sel.addEventListener('mousedown', (ev) => ev.stopPropagation());
    // click 也必须拦住：否则冒泡到 .ann-box 的选中逻辑，会重建 DOM 把
    // <select> 整个销毁掉，原生下拉弹层会立刻被收回（点不开）
    sel.addEventListener('click', (ev) => ev.stopPropagation());
    sel.addEventListener('dblclick', (ev) => ev.stopPropagation());

    // 删除按钮
    const del = document.createElement('button');
    del.className = 'ann-del';
    del.type = 'button';
    del.textContent = '×';
    del.title = '删除这个标注';
    del.addEventListener('click', (ev) => {
      ev.stopPropagation();
      ANN.dets.splice(i, 1);
      ANN.active = -1;
      ANN.dirty = true;
      paintAnn();
    });
    del.addEventListener('mousedown', (ev) => ev.stopPropagation());

    node.appendChild(sel);
    node.appendChild(del);

    if (ANN.shape === 'dot') {
      // 点模式：只在框中心画一个点，框用虚线淡显（hover 才浮现）
      node.classList.add('as-dot');
      const dot = document.createElement('i');
      dot.className = 'ann-dot';
      // 必须用 50%/50%，不能用 pct(cx, naturalW)：
      // .ann-dot 是 .ann-box 的子元素，而 .ann-box 是 position:absolute，
      // 它的百分比基准是**框**（不是图片）—— 写成 pct(cx, 图宽) 会让点
      // 落在框内 44% 而不是 50%，即偏在框心左侧（框越窄偏得越多），
      // 抓起来拖动就会感觉“不跟手”。框本来就由 cx/cy 算出，框心即中心点。
      dot.style.left = '50%';
      dot.style.top = '50%';
      node.appendChild(dot);
    } else {
      // 框模式：四个角把手用于缩放
      ['nw', 'ne', 'sw', 'se'].forEach((pos) => {
        const h = document.createElement('span');
        h.className = `ann-handle ${pos}`;
        h.dataset.handle = pos;
        node.appendChild(h);
      });
    }

    node.addEventListener('mousedown', (ev) => startAnnDrag(ev, i, node, 'move'));
    node.addEventListener('click', (ev) => {
      ev.stopPropagation();
      ANN.active = i;
      syncAnnActive();   // 只切高亮，不重建 DOM（重建会打断下拉框等交互）
    });

    wrap.appendChild(node);
  });

  el('annCount').textContent = `${visible.length}/${ANN.dets.length}`;
  el('annDeleteAll').disabled = !ANN.dets.length;
  el('annSave').textContent = ANN.dirty ? '保存 *' : '保存';
}

/** 把 bbox 写成节点的百分比定位（左/上/宽/高）
 *
 * 拖拽过程中只改这一个节点的样式，不重建整个框层 ——
 * 每次 mousemove 都 paintAnn() 会反复销毁/重建 DOM，手感发顿（不跟手）。
 */
function applyAnnBoxRect(node, bbox) {
  const [x1, y1, x2, y2] = bbox;
  const pct = (v, total) => `${(v / total) * 100}%`;
  node.style.left = pct(x1, ANN.naturalW);
  node.style.top = pct(y1, ANN.naturalH);
  node.style.width = pct(x2 - x1, ANN.naturalW);
  node.style.height = pct(y2 - y1, ANN.naturalH);
}

/** 只切「哪个框是选中态」，不重建 DOM */
function syncAnnActive() {
  document.querySelectorAll('#annBoxes .ann-box').forEach((n) => {
    n.classList.toggle('active', Number(n.dataset.i) === ANN.active);
  });
}

/** 在图片上拖拽：移动已有框 / 拖角缩放 / 空白处画新框 */
function startAnnDrag(ev, i, node, mode) {
  // 点在下拉框或删除按钮上时不要进入拖拽（它们有自己的交互）
  if (ev.target && ev.target.closest && ev.target.closest('.ann-cls-select, .ann-del')) {
    return;
  }
  ev.preventDefault();
  ev.stopPropagation();

  const rect = el('annCanvas').getBoundingClientRect();
  const sx = ANN.naturalW / rect.width;
  const sy = ANN.naturalH / rect.height;
  const startX = (ev.clientX - rect.left) * sx;
  const startY = (ev.clientY - rect.top) * sy;
  const handle = ev.target.closest('.ann-handle');
  const box = ANN.dets[i];
  const orig = box.bbox.slice();

  const onMove = (e) => {
    const mx = (e.clientX - rect.left) * sx;
    const my = (e.clientY - rect.top) * sy;
    const dx = mx - startX;
    const dy = my - startY;

    if (handle && mode === 'move') {
      // 拖角缩放
      const pos = handle.dataset.handle;
      let [x1, y1, x2, y2] = orig;
      if (pos.includes('w')) x1 = Math.min(orig[0] + dx, x2 - 2);
      if (pos.includes('e')) x2 = Math.max(orig[2] + dx, x1 + 2);
      if (pos.includes('n')) y1 = Math.min(orig[1] + dy, y2 - 2);
      if (pos.includes('s')) y2 = Math.max(orig[3] + dy, y1 + 2);
      box.bbox = [clamp(x1, 0, ANN.naturalW), clamp(y1, 0, ANN.naturalH), clamp(x2, 0, ANN.naturalW), clamp(y2, 0, ANN.naturalH)];
    } else {
      // 整体移动
      const w = orig[2] - orig[0];
      const h = orig[3] - orig[1];
      const nx = clamp(orig[0] + dx, 0, ANN.naturalW - w);
      const ny = clamp(orig[1] + dy, 0, ANN.naturalH - h);
      box.bbox = [nx, ny, nx + w, ny + h];
    }
    ANN.dirty = true;
    // 只更新被拖的这个节点，不重建整个框层
    applyAnnBoxRect(node, box.bbox);
  };

  const onUp = () => {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    paintAnn();   // 松手后再整体重画一次，把标签/计数等同步过来
  };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

/** 空白处按下 → 画新框 */
function startAnnDraw(ev) {
  if (!ANN || ev.target.closest('.ann-box')) return;
  // 关键：阻止 mousedown 的默认行为，否则浏览器会接管成
  // 「拖拽图片」或「选中文字」，与长按画框的手势冲突
  ev.preventDefault();
  const rect = el('annCanvas').getBoundingClientRect();
  const sx = ANN.naturalW / rect.width;
  const sy = ANN.naturalH / rect.height;
  const x0 = clamp((ev.clientX - rect.left) * sx, 0, ANN.naturalW);
  const y0 = clamp((ev.clientY - rect.top) * sy, 0, ANN.naturalH);

  const temp = document.createElement('div');
  temp.className = 'ann-box drawing';
  el('annBoxes').appendChild(temp);

  const onMove = (e) => {
    const x1 = clamp((e.clientX - rect.left) * sx, 0, ANN.naturalW);
    const y1 = clamp((e.clientY - rect.top) * sy, 0, ANN.naturalH);
    temp.style.left = `${(Math.min(x0, x1) / ANN.naturalW) * 100}%`;
    temp.style.top = `${(Math.min(y0, y1) / ANN.naturalH) * 100}%`;
    temp.style.width = `${(Math.abs(x1 - x0) / ANN.naturalW) * 100}%`;
    temp.style.height = `${(Math.abs(y1 - y0) / ANN.naturalH) * 100}%`;
  };
  onMove(ev);

  const onUp = (e) => {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    temp.remove();

    const x1 = clamp((e.clientX - rect.left) * sx, 0, ANN.naturalW);
    const y1 = clamp((e.clientY - rect.top) * sy, 0, ANN.naturalH);
    const bx1 = Math.min(x0, x1);
    const by1 = Math.min(y0, y1);
    const bx2 = Math.max(x0, x1);
    const by2 = Math.max(y0, y1);
    // 太小的框视为误触，丢弃
    if (bx2 - bx1 < 4 || by2 - by1 < 4) return;

    const cid = ANN.filter !== null ? ANN.filter : 0;
    ANN.dets.push({
      class_id: cid,
      class_name: ANN.dsClasses[cid] || String(cid),
      confidence: null,
      bbox: [bx1, by1, bx2, by2],
    });
    ANN.active = ANN.dets.length - 1;
    ANN.dirty = true;
    paintAnn();
  };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(v, hi));
}

el('annCanvas').addEventListener('mousedown', startAnnDraw);

/** 显示形式切换：框 / 中心点 */
document.querySelectorAll('#annShape .seg-btn').forEach((b) => {
  b.addEventListener('click', () => {
    if (!ANN) return;
    ANN.shape = b.dataset.shape || 'box';
    document
      .querySelectorAll('#annShape .seg-btn')
      .forEach((x) => x.classList.toggle('active', x === b));
    paintAnn();
  });
});

/** 标签开关：关掉后任何模式下都不显示类别标签（点模式 hover 也只出框） */
el('annShowLabels').addEventListener('change', (ev) => {
  if (!ANN) return;
  ANN.showLabels = ev.target.checked;
  el('annBoxes').classList.toggle('no-labels', !ANN.showLabels);
});

/** 预标注：调后端推理，把结果直接变成标注框 */
el('annPrelabel').addEventListener('click', async () => {
  if (!ANN) return;
  const btn = el('annPrelabel');
  btn.disabled = true;
  annMsg('预标注中…（首次要加载模型，可能较慢）');
  try {
    const { weights, classes } = readDetectOptions('ann');
    const q = new URLSearchParams({
      conf: el('annConf').value || '0.38',
      weights,
      imgsz: '640',
    });
    if (classes) q.set('classes', classes);

    const res = await api(
      `/api/annotate/prelabel/${ANN.rel}?${q}`,
      { method: 'POST' },
      300000
    );
    const dets = (res.detections || []).map((d) => ({
      class_id: d.class_id,
      class_name: d.class_name,
      confidence: d.confidence,
      bbox: d.bbox.slice(),
    }));
    // 预标注可能给数据集追加了新类别（如 COCO 的 chair），刷新左侧类别列表
    if (Array.isArray(res.classes) && res.classes.length) {
      ANN.dsClasses = res.classes;
      ANN.filter = null;
      paintAnnClasses();
    }
    ANN.dets = dets;
    ANN.active = -1;
    ANN.dirty = true;
    paintAnn();
    annMsg(`预标注完成：${dets.length} 个目标（改完记得保存）`);
  } catch (e) {
    annMsg(`预标注失败：${e instanceof Error ? e.message : String(e)}`);
  } finally {
    btn.disabled = false;
  }
});

/** 清空全部标注 */
el('annDeleteAll').addEventListener('click', () => {
  if (!ANN || !ANN.dets.length) return;
  if (!confirm(`清空这 ${ANN.dets.length} 个标注？`)) return;
  ANN.dets = [];
  ANN.active = -1;
  ANN.dirty = true;
  paintAnn();
});

/** 保存为 YOLO label */
el('annSave').addEventListener('click', async () => {
  if (!ANN) return;
  const btn = el('annSave');
  btn.disabled = true;
  annMsg('保存中…');
  try {
    const res = await api(
      `/api/annotate/save/${ANN.rel}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          detections: ANN.dets.map((d) => ({ class_id: d.class_id, bbox: d.bbox })),
        }),
      },
      60000
    );
    ANN.dirty = false;
    paintAnn();
    annMsg(`已保存 ${res.count} 个标注 → ${res.label}`);
    loadAnnotateGrid();   // 刷新网格里的已标注数量
  } catch (e) {
    annMsg(`保存失败：${e instanceof Error ? e.message : String(e)}`);
  } finally {
    btn.disabled = false;
  }
});

/** 关闭标注器（有未保存改动时提示） */
function closeAnnotator() {
  if (ANN && ANN.dirty && !confirm('有未保存的改动，确定关闭吗？')) return;
  el('annotatorMask').hidden = true;
  el('annBaseImg').src = '';
  ANN = null;
}
el('annClose').addEventListener('click', closeAnnotator);
el('annotatorMask').addEventListener('click', (ev) => {
  // 点遮罩空白处 = 关闭；点标注器内部不关
  if (ev.target === el('annotatorMask')) closeAnnotator();
});
document.addEventListener('keydown', (ev) => {
  if (!ANN) return;
  const t = ev.target;
  const tag = t && t.tagName ? String(t.tagName).toUpperCase() : '';
  // 焦点在输入框/下拉框里时，键盘事件交给控件自己处理
  const typing = tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA';
  if (ev.key === 'Escape') {
    if (typing) return;      // 只收起下拉框，不要连带关掉整个标注器
    closeAnnotator();
    return;
  }
  // 选中标注后按 Delete 删除（避免在控件里打字时误删）
  if (ev.key === 'Delete' && ANN.active >= 0) {
    if (typing) return;
    ev.preventDefault();
    ANN.dets.splice(ANN.active, 1);
    ANN.active = -1;
    ANN.dirty = true;
    paintAnn();
  }
});

/* ==================== 通用：轮询后台任务 ====================
 * - 单次查询 15 秒超时
 * - 偶发失败自动重试（连续 5 次才判失败）
 * - 一直显示已用时，避免"看起来卡死"
 */
function pollJob(jobId, box, onFinish, t0 = Date.now()) {
  let failCount = 0;

  /** 取消按钮（点击调取消接口，失败也提示） */
  const cancelBtn = () =>
    `<button class="btn small danger" data-cancel="${esc(jobId)}">取消任务</button>`;

  // 给 box 绑定取消事件（委托，避免反复重建）
  if (!box._cancelBound) {
    box._cancelBound = true;
    box.addEventListener('click', async (ev) => {
      const b = ev.target.closest('[data-cancel]');
      if (!b) return;
      b.disabled = true;
      b.textContent = '取消中…';
      try {
        await api(`/api/jobs/${encodeURIComponent(b.dataset.cancel)}/cancel`, { method: 'POST' }, 15000);
      } catch (e) {
        banner(`取消失败：${e instanceof Error ? e.message : String(e)}`);
        b.disabled = false;
        b.textContent = '取消任务';
      }
    });
  }

  const timer = setInterval(async () => {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    try {
      const job = await api(`/api/jobs/${jobId}`, {}, 15000);
      failCount = 0;

      const pct = job.total ? Math.round((job.progress / job.total) * 100) : 0;
      box.innerHTML =
        `<div>${esc(job.message)}</div>` +
        `<div class="progress"><div style="width:${pct}%"></div></div>` +
        `<div style="font-size:12px;color:#80868b">` +
        `${esc(job.progress)}/${esc(job.total || '?')}（${pct}%）` +
        `｜已用时 <span class="elapsed">${elapsed}s</span></div>` +
        ` <div style="margin-top:8px">${cancelBtn()}</div>`;

      if (job.status !== 'running') {
        clearInterval(timer);
        onFinish(job);
      }
    } catch (err) {
      failCount += 1;
      console.warn(`[pollJob] 第 ${failCount} 次状态查询失败`, err);
      if (failCount >= 5) {
        clearInterval(timer);
        showError(box, err, `任务状态查询连续失败 ${failCount} 次（任务 ID：${jobId}）`);
      } else {
        box.innerHTML =
          `<div>任务进行中…（状态查询临时失败 ${failCount}/5，自动重试中）</div>` +
          `<div style="font-size:12px;color:#80868b">已用时 <span class="elapsed">${elapsed}s</span></div>` +
          ` <div style="margin-top:8px">${cancelBtn()}</div>`;
      }
    }
  }, 1500);
}

/* ==================== 初始化 ==================== */
loadEnv();
loadImages();
