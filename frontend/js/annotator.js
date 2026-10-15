import { el, esc, api, banner, showError } from './core.js?v=82';
import { readDetectOptions } from './models.js?v=82';
import { classColor, classColorA, createPoseLayer } from './viewer.js?v=82';
import { loadDatasets } from './training.js?v=82';

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
    el('behaviorExport').hidden = !data.behavior;
    el('behaviorExport').href = `/api/behavior-datasets/${encodeURIComponent(ds)}/export`;
    const sub = (data.subsets && data.subsets[split]) || null;
    if (!sub || !sub.count) {
      grid.innerHTML = `<div class="empty">该子集没有图片（${esc(ds)}/images/${esc(split)}）</div>`;
      stat.textContent = '';
      return;
    }
    stat.textContent = `${sub.count} 张 · 已标注 ${sub.labeled} 张 · ${(data.classes || []).length} 类`;
    if (data.behavior) stat.textContent += `｜行为已标 ${data.behavior.labeled}/${data.behavior.total} 人（举手 ${data.behavior.counts['举手']}）`;

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
    behavior: false, pose: null,
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
    ANN.behavior = !!data.behavior;
    ANN.pose = data.pose || null;
    el('annPrelabel').disabled = ANN.behavior;
    ANN.dets = (data.detections || []).map((d) => ({
      ...d,
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
    annMsg(ANN.behavior ? '每人选一个状态。举手已自动预标，可人工纠错。选中框后：1 低头 / 2 抬头 / 3 正常 / 4 无法判断 / 5 举手。改完点保存。' : '');
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
    if (!ANN.behavior) row.appendChild(ed);
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
  if (!ANN.behavior) box.appendChild(add);
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
    if (!ANN.behavior) sel.appendChild(rn);
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

  if (ANN.pose) wrap.appendChild(createPoseLayer(visible.map(i => ANN.dets[i]), ANN.pose, ANN.naturalW, ANN.naturalH));
  const list = el('annList'); list.innerHTML = '';
  if (ANN.behavior) visible.forEach(i => {
    const button = document.createElement('button');
    button.className = 'btn small ghost'; button.textContent = `#${i+1} ${ANN.dets[i].class_name}`;
    button.addEventListener('click', () => { ANN.active = i; syncAnnActive(); });
    list.appendChild(button);
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
    const { weights, conf, classes, headDetection, rtdetrBase, mergeIou, mergeConfidence, headDownRThreshold, headUpRThreshold, poseRoi, roiConfidence, isPose } = readDetectOptions('ann');
    const q = new URLSearchParams({
      conf,
      weights,
      imgsz: '640',
    });
    if (classes) q.set('classes', classes);
    if (headDetection) q.set('head_detection', 'true');
    if (isPose) { q.set('head_down_r_threshold', String(headDownRThreshold)); q.set('head_up_r_threshold', String(headUpRThreshold)); q.set('pose_roi', String(poseRoi)); q.set('roi_confidence', String(roiConfidence)); }
    if (headDetection) q.set('merge_iou', String(mergeIou));
    if (headDetection) q.set('merge_confidence', String(mergeConfidence));
    if (rtdetrBase) q.set('rtdetr_base', 'true');

    const res = await api(
      `/api/annotate/prelabel/${ANN.rel}?${q}`,
      { method: 'POST' },
      300000
    );
    if (headDetection && !res.fusion) throw new Error('目标检测未生效，请重启服务后刷新页面重试。');
    if (rtdetrBase && !res.fusion?.rtdetr_base) throw new Error('RT-DETR 基准未生效，请重启服务后刷新页面。');
    if (headDetection && Number(res.fusion?.merge_iou ?? 0) !== mergeIou) throw new Error('合并 IoU 阈值未生效，请重启服务后刷新页面。');
    if (headDetection && Number(res.fusion?.merge_confidence ?? 0) !== mergeConfidence) throw new Error('合并置信度门槛未生效，请重启服务后刷新页面。');
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
    annMsg(`预标注完成${headDetection ? '（目标检测融合）' : ''}：${dets.length} 个目标（改完记得保存）`);
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
          detections: ANN.dets.map((d) => ({ class_id: d.class_id, bbox: d.bbox, sample_id: d.sample_id })),
        }),
      },
      60000
    );
    ANN.dirty = false;
    if (Array.isArray(res.detections)) ANN.dets = res.detections;
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
  if (ANN.behavior && !typing && ANN.active >= 0 && /^[1-5]$/.test(ev.key)) {
    const det = ANN.dets[ANN.active];
    det.class_id = Number(ev.key); det.class_name = ANN.dsClasses[det.class_id];
    ANN.dirty = true; ev.preventDefault(); paintAnn(); return;
  }
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


async function importBehaviorRecords(recordIds = null) {
  const payload = { dataset: el('behaviorDatasetName').value.trim() || 'east_classroom_behavior' };
  if (recordIds) payload.record_ids = recordIds;
  const result = await api('/api/behavior-datasets/import', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }, 60000);
  await loadDatasets();
  el('annotateDataset').value = result.dataset; el('annotateSplit').value = 'train';
  document.querySelector('.tab[data-tab="dataset"]').click();
  await loadAnnotateGrid();
  el('behaviorImportStatus').textContent = `新增 ${result.imported} 张，已存在 ${result.skipped} 张；共 ${result.progress.total} 人。`;
  return result;
}
el('behaviorImportEast').addEventListener('click', async () => {
  const button = el('behaviorImportEast'); button.disabled = true;
  try { await importBehaviorRecords(); }
  catch (error) { banner(error instanceof Error ? error.message : String(error)); }
  finally { button.disabled = false; }
});

export { loadAnnotateGrid, importBehaviorRecords };
