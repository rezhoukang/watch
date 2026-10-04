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
    if (name === 'train') loadDatasets();
  });
});

/* ==================== 环境状态 ==================== */

/* ==================== 环境状态 / 模型模式（普通 YOLO ↔ YOLO-World）==================== */

let WORLD_MODELS = [];   // 属于 YOLO-World 的权重文件名（由后端告知）
let DEFAULT_MODEL = 'yolov8s-worldv2.pt';   // 下拉框默认选中（后端 status 覆盖）

/** 填充下拉框，并尽量保留原选择；无选择时用 preferred（默认模型） */
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
  el(wrapId).hidden = !WORLD_MODELS.includes(name);
}

async function loadEnv() {
  const badge = el('envStatus');
  try {
    const s = await api('/api/yolo/status');
    badge.textContent = s.message;
    badge.className = 'badge ' + (s.available ? 'ok' : 'err');

    WORLD_MODELS = s.world_models || [];
    DEFAULT_MODEL = s.default_model || DEFAULT_MODEL;

    const all = s.models.length ? s.models : [DEFAULT_MODEL];
    fillSelect('detWeights', all, DEFAULT_MODEL);   // 图片检测：全部模型
    fillSelect('vidWeights', all, DEFAULT_MODEL);   // 视频检测：全部模型
    // 训练：排除 World 权重（训练走普通 YOLO 流程，喂 World 权重会失败）
    fillSelect('trainWeights', all.filter((m) => !WORLD_MODELS.includes(m)), DEFAULT_MODEL);

    syncModeUi('detWeights', 'detClassesWrap');
    syncModeUi('vidWeights', 'vidClassesWrap');
  } catch (e) {
    badge.textContent = '状态获取失败：' + (e instanceof Error ? e.message : String(e));
    badge.className = 'badge err';
  }
}

// 切换模型 → 立即更新 prompt 输入框的显隐
el('detWeights').addEventListener('change', () =>
  syncModeUi('detWeights', 'detClassesWrap')
);
el('vidWeights').addEventListener('change', () =>
  syncModeUi('vidWeights', 'vidClassesWrap')
);

/** 读取当前检测设置（图片/视频共用） */
function readDetectOptions(prefix) {
  const weights = el(`${prefix}Weights`).value || DEFAULT_MODEL;
  const isWorld = WORLD_MODELS.includes(weights);
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

/* ==================== 图片检测：图库 / 结果 两个子页面 ==================== */

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

/* ==================== 检测结果查看器（框由前端渲染）==================== */

let DET = null;   // 当前查看的检测结果

/** 去重默认开关：勾选框与 DET.dedup 的初值都取它，避免「界面勾着但没生效」 */
const DEDUP_DEFAULT = true;

/** 类别 → 颜色：黄金角取色，区分度高且白字可读 */
function classColor(i) {
  return `hsl(${Math.round((i * 137.508) % 360)}, 62%, 42%)`;
}

/** 渲染整个结果页（原图 + 前端叠加框 + 筛选条 + 明细表） */
function renderDetectResult(r, filename, confUsed, elapsed, imgsz) {
  const box = el('detectResult');
  const dets = Array.isArray(r.detections) ? r.detections : [];

  // 类别按出现次数降序，分配固定颜色
  const counts = new Map();
  dets.forEach((d) => counts.set(d.class_name, (counts.get(d.class_name) || 0) + 1));
  const classes = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([name]) => name);
  const colorOf = new Map(classes.map((name, i) => [name, classColor(i)]));

  DET = {
    dets,
    classes,
    colorOf,
    hidden: new Set(),          // 被关掉的类别
    solo: null,                 // 双击胶囊 → 只看这一类
    shape: 'box',               // 图形：'box' 框 | 'dot' 中心点
    confMin: Number(confUsed) || 0.01,
    naturalW: 1,
    naturalH: 1,
    // 去重（后处理，不重跑模型）
    dedup: DEDUP_DEFAULT,
    dedupDist: 20,              // 中心点距离 ≤ 这个像素数 → 视为同一目标
    dedupNest: 0.8,             // 一个框被另一个覆盖 ≥ 这个比例 → 视为同一目标
    dedupKeep: 'conf',          // 冲突时先留谁：'conf' 得分高 / 'big' 大框 / 'small' 小框
  };

  const modeLabel = r.mode === 'world' ? 'YOLO-World' : '普通 YOLO';
  const clsNote =
    r.mode === 'world'
      ? `prompt「${(r.classes_used || []).join(', ') || '权重内置'}」`
      : `内置 ${(r.classes_used || []).length} 类`;

  box.innerHTML =
    `<div class="result-head">${esc(filename)}` +
      `<span class="meta">${esc(modeLabel)}｜${esc(r.model || '')}｜${esc(clsNote)}｜` +
      `检测阈值 ${esc(confUsed)}｜imgsz ${esc(r.imgsz || imgsz)}｜耗时 ${elapsed.toFixed(1)}s</span></div>` +
    `<div class="viewer-bar">` +
      `<div class="seg" id="detShape">` +
        `<button class="seg-btn active" data-shape="box">框</button>` +
        `<button class="seg-btn" data-shape="dot">中心点</button>` +
      `</div>` +
      `<label>置信度 ≥ <input type="range" id="detMinConf" min="0.01" max="0.99" step="0.01" value="${esc(confUsed)}">` +
      `<b id="detMinConfVal">${esc(confUsed)}</b></label>` +
      `<label><input type="checkbox" id="detAlwaysTag"> 常显标签</label>` +
      `<div class="chips" id="detChips"></div>` +
    `</div>` +
    `<div class="dedup-bar">` +
      `<label><input type="checkbox" id="dedupOn"${DEDUP_DEFAULT ? ' checked' : ''}> 去重</label>` +
      `<label>中心距 ≤ <input type="number" id="dedupDist" value="20" min="0" max="500" step="1"> px</label>` +
      `<label>内嵌率 ≥ <input type="number" id="dedupNest" value="80" min="0" max="100" step="5"> %</label>` +
      `<label>重叠时保留 <select id="dedupKeep">` +
        `<option value="conf">得分高</option>` +
        `<option value="big">大框（外包）</option>` +
        `<option value="small">小框（内嵌）</option>` +
      `</select></label>` +
    `</div>` +
    `<div class="viewer" id="detViewer">` +
      `<img class="base" id="detBaseImg" src="/images/${encodeURIComponent(filename)}" alt="检测图">` +
      `<div class="boxes" id="detBoxes"></div>` +
    `</div>` +
    `<div class="viewer-foot">` +
      `<span id="detCount"></span>` +
      (r.result_url ? `<a href="${esc(r.result_url)}" target="_blank">后端渲染图（对照）</a>` : '') +
    `</div>` +
    `<div class="det-table-wrap"><table>` +
      `<thead><tr><th>类别</th><th>置信度</th><th>中心 (cx, cy)</th><th>宽 × 高</th>` +
      `<th>左上右下 (x1, y1, x2, y2)</th></tr></thead>` +
      `<tbody id="detTableBody"></tbody></table></div>`;

  // 类别胶囊：单击开关，双击只看这一类
  const chips = el('detChips');
  if (classes.length) {
    classes.forEach((name) => {
      const b = document.createElement('button');
      b.className = 'chip';
      b.dataset.name = name;
      b.style.background = colorOf.get(name);
      b.textContent = `${name} ${counts.get(name)}`;   // 初值；paintDetects 会按实际显示数刷新
      b.title = '单击：显示 / 隐藏这一类｜双击：只看这一类';
      b.addEventListener('click', () => {
        if (DET.hidden.has(name)) DET.hidden.delete(name);
        else DET.hidden.add(name);
        DET.solo = null;
        syncChipState();
        paintDetects();
      });
      b.addEventListener('dblclick', () => {
        const soloing = DET.solo === name;
        DET.solo = soloing ? null : name;
        DET.hidden = soloing ? new Set() : new Set(classes.filter((n) => n !== name));
        syncChipState();
        paintDetects();
      });
      chips.appendChild(b);
    });
  } else {
    chips.innerHTML = '<span class="empty-inline">未检出任何目标（可调低检测阈值重跑）</span>';
  }

  // 明细表一次建好，靠 hidden 切换（不重建，避免闪烁）
  const tbody = el('detTableBody');
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
    tr.addEventListener('mouseenter', () => highlightDet(i, true));
    tr.addEventListener('mouseleave', () => highlightDet(i, false));
    tbody.appendChild(tr);
  });

  // 显示方式：框 / 中心点
  document.querySelectorAll('#detShape .seg-btn').forEach((b) => {
    b.addEventListener('click', () => {
      DET.shape = b.dataset.shape || 'box';
      document
        .querySelectorAll('#detShape .seg-btn')
        .forEach((x) => x.classList.toggle('active', x === b));
      paintDetects();
    });
  });

  // 二次筛选：不用重跑模型，纯前端过滤
  const slider = el('detMinConf');
  slider.addEventListener('input', () => {
    DET.confMin = Number(slider.value);
    el('detMinConfVal').textContent = DET.confMin.toFixed(2);
    paintDetects();
  });
  el('detAlwaysTag').addEventListener('change', (e) => {
    el('detViewer').classList.toggle('always-tags', e.target.checked);
  });

  // 去重：纯前端后处理，参数随时可调
  const applyDedup = () => {
    DET.dedup = el('dedupOn').checked;
    DET.dedupDist = Number(el('dedupDist').value) || 0;
    DET.dedupNest = (Number(el('dedupNest').value) || 0) / 100;
    DET.dedupKeep = el('dedupKeep').value;
    paintDetects();
  };
  el('dedupOn').addEventListener('change', applyDedup);
  el('dedupKeep').addEventListener('change', applyDedup);
  ['dedupDist', 'dedupNest'].forEach((id) => el(id).addEventListener('input', applyDedup));

  // 图片就绪后再画框（需要原图宽高把 bbox 换算成百分比）
  const img = el('detBaseImg');
  const onReady = () => {
    DET.naturalW = img.naturalWidth || 1;
    DET.naturalH = img.naturalHeight || 1;
    paintDetects();
  };
  if (img.complete && img.naturalWidth) onReady();
  else img.addEventListener('load', onReady);
  img.addEventListener('error', () => {
    el('detCount').textContent = `原图加载失败：/images/${filename}`;
  });
}

/** 让胶囊的亮/灰状态与 DET.hidden 保持一致 */
function syncChipState() {
  document.querySelectorAll('#detChips .chip').forEach((b) => {
    b.classList.toggle('off', DET.hidden.has(b.dataset.name));
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
 * 返回保留下来的原始下标（升序）。
 */
function dedupPick(list) {
  const rank = {
    conf: (d) => -d.confidence,
    big: (d) => -detArea(d),
    small: (d) => detArea(d),
  }[DET.dedupKeep];

  const sorted = [...list].sort((i, j) => rank(DET.dets[i]) - rank(DET.dets[j]));
  const kept = [];

  for (const i of sorted) {
    const a = DET.dets[i];
    let dup = false;
    for (const k of kept) {
      const b = DET.dets[k];
      if (a.class_name !== b.class_name) continue;   // 跨类别不合并（人身上可以有手机）
      if (DET.dedupDist > 0 && centerDist(a, b) <= DET.dedupDist) {
        dup = true;
        break;
      }
      if (
        DET.dedupNest > 0 &&
        (coveredRatio(a, b) >= DET.dedupNest || coveredRatio(b, a) >= DET.dedupNest)
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
function bindDetNode(node, i) {
  node.addEventListener('mouseenter', () => highlightDet(i, true));
  node.addEventListener('mouseleave', () => highlightDet(i, false));
  node.addEventListener('click', () => {
    const tr = document.querySelector(`#detTableBody tr[data-i="${i}"]`);
    if (tr) tr.scrollIntoView({ block: 'nearest' });
  });
}

/** 按当前筛选条件重画标注（框 或 中心点），并同步明细表 */
function paintDetects() {
  if (!DET) return;
  const wrap = el('detBoxes');
  wrap.innerHTML = '';

  const show = [];
  DET.dets.forEach((d, i) => {
    if (DET.hidden.has(d.class_name)) return;
    if (d.confidence < DET.confMin) return;
    show.push(i);
  });

  // 去重是最后一道：先按类别/置信度筛完，再合并重复目标
  const before = show.length;
  const shown = DET.dedup ? dedupPick(show) : show;
  const merged = before - shown.length;

  shown.forEach((i) => {
    const d = DET.dets[i];
    const [x1, y1, x2, y2] = d.bbox;
    // 类别色只写一次：CSS 里边框 / 圆点 / 标签底色都取 var(--c)
    const color = DET.colorOf.get(d.class_name) || '#1a73e8';
    const pct = (v, total) => `${(v / total) * 100}%`;

    if (DET.shape === 'dot') {
      // 1) 幽灵框：平时隐藏，hover 这个点时才浮现，用来看清它到底框住多大范围
      const ghost = document.createElement('div');
      ghost.className = 'bbox ghost';
      ghost.dataset.i = String(i);
      ghost.style.setProperty('--c', color);
      ghost.style.left = pct(x1, DET.naturalW);
      ghost.style.top = pct(y1, DET.naturalH);
      ghost.style.width = pct(x2 - x1, DET.naturalW);
      ghost.style.height = pct(y2 - y1, DET.naturalH);
      wrap.appendChild(ghost);

      // 2) 点（放在框后面 = 显示在框上面）
      const [cx, cy] = detCenter(d);
      const dot = document.createElement('div');
      dot.className = 'dot';
      dot.dataset.i = String(i);
      dot.style.setProperty('--c', color);
      dot.style.left = pct(cx, DET.naturalW);
      dot.style.top = pct(cy, DET.naturalH);
      dot.appendChild(document.createElement('i'));
      dot.appendChild(makeDetTag(d));
      bindDetNode(dot, i);
      wrap.appendChild(dot);
    } else {
      const node = document.createElement('div');
      node.className = 'bbox';
      node.dataset.i = String(i);
      node.style.setProperty('--c', color);
      node.style.left = pct(x1, DET.naturalW);
      node.style.top = pct(y1, DET.naturalH);
      node.style.width = pct(x2 - x1, DET.naturalW);
      node.style.height = pct(y2 - y1, DET.naturalH);
      node.appendChild(makeDetTag(d));
      bindDetNode(node, i);
      wrap.appendChild(node);
    }
  });

  const visible = new Set(shown);
  el('detTableBody').querySelectorAll('tr').forEach((tr) => {
    tr.hidden = !visible.has(Number(tr.dataset.i));
  });

  // 胶囊上的数字 = 图上实际显示的该类数量，跟下面那句"N 个框"对得上（受筛选/去重影响）
  const shownCounts = new Map();
  shown.forEach((i) => {
    const name = DET.dets[i].class_name;
    shownCounts.set(name, (shownCounts.get(name) || 0) + 1);
  });
  document.querySelectorAll('#detChips .chip').forEach((b) => {
    const n = shownCounts.get(b.dataset.name) || 0;
    b.textContent = `${b.dataset.name} ${n}`;
    b.classList.toggle('zero', n === 0);
  });

  const notes = [];
  if (merged > 0) notes.push(`去重合并 ${merged} 个`);
  if (shown.length + merged < DET.dets.length) notes.push('部分被类别/置信度筛选隐藏');

  const unit = DET.shape === 'dot' ? '个点' : '个框';
  el('detCount').textContent =
    `图上 ${shown.length} ${unit} / 共检出 ${DET.dets.length} 个` +
    (notes.length ? `（${notes.join('；')}）` : '');
}

/** 标注 ↔ 明细表行 双向高亮；中心点模式下顺带浮现它对应的框 */
function highlightDet(i, on) {
  document
    .querySelectorAll(`#detBoxes [data-i="${i}"]:not(.ghost)`)
    .forEach((n) => n.classList.toggle('hl', on));
  document
    .querySelectorAll(`#detBoxes .bbox.ghost[data-i="${i}"]`)
    .forEach((n) => n.classList.toggle('show', on));
  const tr = document.querySelector(`#detTableBody tr[data-i="${i}"]`);
  if (tr) tr.classList.toggle('hl', on);
}

/* ==================== 图片检测 ==================== */

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
      fig.title = '点击检测：' + img.filename;
      fig.innerHTML = `<img src="${esc(img.url)}" alt=""><figcaption>${esc(img.filename)}</figcaption>`;

      // 删除按钮（hover 时才露出，点它不会触发检测）
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
    const conf = el('detConf').value || '0.01';
    const imgsz = el('detImgsz').value || '640';
    const { weights, isWorld, classes } = readDetectOptions('det');
    const modeText = isWorld
      ? `YOLO-World${classes ? `｜prompt「${classes}」` : '｜prompt=权重内置'}`
      : '普通 YOLO';

    // 切到「检测结果」子页面，并显示当前图片名
    switchSubPane('result');
    el('detResultName').textContent = filename;

    // 2) 实时计时，区分"在跑"和"卡死"
    const paint = () => {
      box.innerHTML =
        `<div>检测中… <b>${((Date.now() - t0) / 1000).toFixed(1)}s</b></div>` +
        `<div style="font-size:12px;color:#80868b;margin-top:6px">` +
        `图片：${esc(filename)}｜置信度 ${esc(conf)}｜imgsz ${esc(imgsz)}｜${esc(modeText)}<br>` +
        `（第一次检测要加载模型；World 切换类别还要编码文字；超过 300 秒自动报超时）</div>`;
    };
    paint();
    ticker = setInterval(paint, 200);

    // 3) 请求后端（图片检测允许长耗时）
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
    showError(box, err, '检测失败');
  } finally {
    // 无论成功失败都停掉计时器，避免"一直检测中"
    if (ticker !== null) clearInterval(ticker);
  }
}

el('refreshImages').addEventListener('click', loadImages);

/* ==================== 视频检测 ==================== */

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

      // 时间轴：展开后选择检测的时间段
      const timeBtn = document.createElement('button');
      timeBtn.className = 'btn small ghost';
      timeBtn.textContent = '时间轴';
      timeBtn.title = '只检测其中一段（如中间 10 分钟）';
      timeBtn.addEventListener('click', () => toggleVideoRange(item, v.filename));

      const btn = document.createElement('button');
      btn.className = 'btn small';
      btn.textContent = '检测';
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

  // 找到对应的列表项（检测按钮的父级 .item），读它自己的时间段
  const item = btn.closest('.item');

  try {
    const conf = el('vidConf').value || '0.01';
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
            `<div style="color:#f9a825;font-weight:600">已取消检测任务。</div>`;
        } else if (job.result_url) {
          box.innerHTML =
            `<video src="${esc(job.result_url)}" controls></video>` +
            `<div class="summary">${esc(job.message)}</div>` +
            `<a class="btn small" href="${esc(job.result_url)}" download>导出视频</a>`;
        } else {
          box.innerHTML =
            `<div style="color:#c5221f;font-weight:600">视频检测未成功：${esc(job.message)}</div>`;
        }
        loadVideos();
        loadOutputs();
      },
      t0
    );
  } catch (err) {
    btn.disabled = false;
    showError(box, err, '视频检测失败');
  } finally {
    if (ticker !== null) clearInterval(ticker);
  }
}

el('refreshVideos').addEventListener('click', loadVideos);

/* ==================== 检测结果库（data/outputs/）==================== */
async function loadOutputs() {
  const box = el('outputList');
  try {
    const data = await api('/api/videos');
    const outs = data.outputs || [];
    if (!outs.length) {
      box.innerHTML = '<div class="empty">暂无检测结果视频</div>';
      return;
    }
    box.innerHTML = '';
    outs.forEach((o) => {
      const item = document.createElement('div');
      item.className = 'item';
      item.innerHTML = `<span class="name">${esc(o.filename)}</span>`;

      const right = document.createElement('span');
      right.className = 'item-actions';

      // 预览：内嵌 <video>
      const play = document.createElement('button');
      play.className = 'btn small ghost';
      play.textContent = '预览';
      play.addEventListener('click', () => {
        const r = el('videoResult');
        r.innerHTML = `<video src="${esc(o.url)}" controls autoplay></video>`;
      });

      // 导出（下载）
      const dl = document.createElement('button');
      dl.className = 'btn small ghost';
      dl.textContent = '导出';
      dl.addEventListener('click', () => {
        const a = document.createElement('a');
        a.href = o.url;
        a.download = o.filename;
        a.click();
      });

      // 删除
      const del = document.createElement('button');
      del.className = 'btn small danger';
      del.textContent = '删除';
      del.addEventListener('click', async () => {
        if (!confirm(`删除检测结果「${o.filename}」？`)) return;
        try {
          await api(`/api/outputs/${encodeURIComponent(o.filename)}`, { method: 'DELETE' }, 15000);
          await loadOutputs();
        } catch (e) {
          banner(`删除失败：${e instanceof Error ? e.message : String(e)}`);
        }
      });

      right.append(play, dl, del);
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
  try {
    const data = await api('/api/datasets');
    sel.innerHTML = '';
    if (!data.datasets.length) {
      box.innerHTML = '<div class="empty">暂无数据集</div>';
      sel.innerHTML = '<option value="">（无数据集）</option>';
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

      const o = document.createElement('option');
      o.value = d.name;
      o.textContent = d.name;
      sel.appendChild(o);
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
    weights: el('trainWeights').value || DEFAULT_MODEL,
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
