import { el, esc } from './core.js?v=80';

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
  if (box === el('detectResult')) opts.onRecordOpen?.(r.record_id || null);
  const imgUrl = opts.imgUrl || r.source_url || `/images/${encodeURIComponent(filename)}`;
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
    merge: r.merge || null,
    fusion: r.fusion || null,
    pose: r.pose || null,
    showSkeleton: true,
    // 去重（后处理，不重跑模型）
    dedup: DEDUP_DEFAULT,
    dedupDist: 20,              // 中心点距离 ≤ 这个像素数 → 视为同一目标
    dedupNest: 0.8,             // 一个框被另一个覆盖 ≥ 这个比例 → 视为同一目标
    dedupKeep: 'conf',          // 冲突时先留谁：'conf' 得分高 / 'big' 大框 / 'small' 小框
  };
  view.q = (sel) => box.querySelector(sel);
  view.qs = (sel) => [...box.querySelectorAll(sel)];

  view.headCounting = r.mode === 'rtdetrv4' || r.head_count_only;
  const rtdetrBase = !!r.fusion?.rtdetr_base;
  const fusionBase = rtdetrBase ? 'RT-DETRv4' : 'medium';
  const fusionIncoming = rtdetrBase ? 'medium' : 'RT-DETRv4';
  const baseConf = r.parameters?.base?.confidence ?? (rtdetrBase ? 0.2 : 0.45);
  const incomingConf = r.parameters?.incoming?.confidence ?? (rtdetrBase ? 0.45 : 0.2);
  const rtConf = rtdetrBase ? baseConf : incomingConf;
  const modeLabel = r.fusion ? `目标检测（${fusionBase} 基准，medium 0.45 + RT-DETRv4 ${rtConf}，imgsz 640）` : r.mode === 'merged' ? '合并推理结果' : view.headCounting ? 'RT-DETRv4-S C-RADIOv4' : r.pose ? `YOLO Pose 人体骨架｜${r.pose.head_rule?.head_down_r_threshold != null ? `低头 R < ${r.pose.head_rule.head_down_r_threshold}，抬头 R > ${r.pose.head_rule.head_up_r_threshold}` : `R 阈值 ${r.pose.head_rule?.nose_shoulder_threshold ?? 0.13}`}${r.pose.exclude_unknown ? '｜已排除无法判断者' : ''}` : (r.mode === 'world' ? 'YOLO-World' : '普通 YOLO');
  const clsNote =
    r.mode === 'world'
      ? `prompt「${(r.classes_used || []).join(', ') || '权重内置'}」`
      : (view.headCounting ? '按人头统计人数' : `内置 ${(r.classes_used || []).length} 类`);

  box.innerHTML =
    `<div class="result-head">${esc(filename)}` +
      `<span class="meta">${esc(modeLabel)}｜${esc(r.weight || '')}｜${esc(clsNote)}｜` +
      `${r.fusion ? `推理阈值：${fusionBase} 基准 ${esc(baseConf)} / ${fusionIncoming} 补充 ${esc(incomingConf)}` : `推理阈值 ${esc(confUsed)}`}｜imgsz ${esc(r.imgsz || imgsz)}｜耗时 ${elapsed.toFixed(1)}s</span></div>` +
    `<div class="viewer-bar">` +
      `<div class="seg" data-role="shape">` +
        `<button class="seg-btn active" data-shape="box">框</button>` +
        `<button class="seg-btn" data-shape="fill">实心</button>` +
        `<button class="seg-btn" data-shape="dot">中心点</button>` +
      `</div>` +
      `<label>置信度 ≥ <input type="range" data-role="minconf" min="0.01" max="0.99" step="0.01" value="${esc(confUsed)}">` +
      `<b data-role="minconf-val">${esc(confUsed)}</b></label>` +
      `<label><input type="checkbox" data-role="alwaystag"> 常显标签</label>` +
      (view.pose ? `<label><input type="checkbox" data-role="skeleton" checked> 上半身骨架（关键点 ≥ ${esc(view.pose.keypoint_confidence)}）</label>` : '') +
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
    (r.merge ? `<div class="merge-note">基准 #${esc(r.merge.base_id.slice(0, 8))}（绿框） ← 待合并 #${esc(r.merge.incoming_id.slice(0, 8))}（橙框）｜合并 IoU ≤ ${esc(r.merge.merge_iou ?? 0)}｜新增 ${esc(r.merge.added_count)} 个，超过阈值舍弃 ${esc(r.merge.discarded_overlaps)} 个</div>` : '') +
    (r.fusion ? `<div class="merge-note">${fusionBase} 基准 ${esc(r.fusion.base_count)} 个（绿框）｜${fusionIncoming} 新增 ${esc(r.fusion.added_count)} 个（橙框），超过阈值舍弃 ${esc(r.fusion.discarded_overlaps)} 个。合并 IoU ≤ ${esc(r.fusion.merge_iou ?? 0)}；超过阈值时保留基准框，跨类别同样适用。</div>` : '') +
    (r.merge || r.fusion ? `<div class="merge-note">以上自定义 IoU 仅适用于待补入框置信度 &gt; ${esc((r.merge || r.fusion).merge_confidence ?? 0)}；低于或等于该置信度时，仍要求 IoU=0。</div>` : '') +
    `<div class="viewer" data-role="viewer">` +
      `<img class="base" data-role="baseimg" src="${esc(imgUrl)}" alt="推理图">` +
      `<div class="boxes" data-role="boxes"></div>` +
    `</div>` +
    `<div class="viewer-foot">` +
      `<span data-role="count"></span>` +
      (r.result_url ? `<a href="${esc(r.result_url)}" target="_blank">后端渲染图（对照）</a>` : '') +
      (r.record_url ? `<a href="${esc(r.record_url)}" download="${esc(r.record_id)}.json">下载推理信息 JSON</a>` : '') +
      (r.record_id && opts.onRecordDelete ? `<button class="btn small danger" data-role="delete-record">删除此推理记录</button>` : '') +
    `</div>` +
    `<div class="det-table-wrap"><table>` +
      `<thead><tr><th>类别</th><th>置信度</th><th>中心 (cx, cy)</th><th>宽 × 高</th>` +
      `<th>左上右下 (x1, y1, x2, y2)</th></tr></thead>` +
      `<tbody data-role="tablebody"></tbody></table></div>`;
  if (r.record_id && opts.onRecordDelete) view.q('[data-role="delete-record"]').addEventListener('click', () => opts.onRecordDelete(r.record_id));

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
      `<td>${esc(d.class_name)}${d.behavior?.state ? ` · ${esc(d.behavior.state)}` : ''}</td><td>${esc(d.confidence)}</td>` +
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
  if (view.pose) view.q('[data-role="skeleton"]').addEventListener('change', (e) => {
    view.showSkeleton = e.target.checked;
    paintDetects(view);
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
  tag.textContent = `${d.class_name} ${Number(d.confidence).toFixed(2)}${d.behavior?.state ? ` · ${d.behavior.state}` : ''}`;
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
function createPoseLayer(detections, pose, width, height) {
  const overlay = document.createElement('div');
  overlay.className = 'pose-layer';
  const parts = [];
  const valid = (p) => Array.isArray(p) && p.length >= 3 && p.every(Number.isFinite) &&
    p[2] >= pose.keypoint_confidence && p[0] > 0 && p[0] < width && p[1] > 0 && p[1] < height;
  detections.forEach((det) => {
    const points = det.keypoints || [];
    (pose.skeleton || []).forEach(([a, b]) => {
      if (a < 11 && b < 11 && valid(points[a]) && valid(points[b])) parts.push(`<line x1="${points[a][0]}" y1="${points[a][1]}" x2="${points[b][0]}" y2="${points[b][1]}"/>`);
    });
    points.slice(0, 11).filter(valid).forEach(([x, y]) => parts.push(`<circle cx="${x}" cy="${y}" r="2"/>`));
    const roi = det.behavior?.roi;
    const polygon = roi?.polygon;
    if (Array.isArray(polygon) && polygon.length >= 3 && polygon.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))) {
      parts.push(`<polygon points="${polygon.map(p => p.join(',')).join(' ')}" fill="none" stroke="#ff9900" stroke-width="1"/>`);
    }
    (roi?.objects || []).forEach(obj => {
      const box = obj.bbox;
      if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) return;
      const [x1, y1, x2, y2] = box;
      parts.push(`<rect x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" fill="none" stroke="#ffcc00" stroke-width="2"/><text x="${x1}" y="${Math.max(10, y1-3)}" fill="#b06b00" font-size="12">${esc(obj.class_name)} ${esc(obj.confidence)}</text>`);
    });
  });
  overlay.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="人体姿态骨架">${parts.join('')}</svg>`;
  return overlay;
}

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
    const color = view.fusion ? (d.source_model === view.fusion.incoming_weight ? '#ff9900' : '#00bb66') : view.merge
      ? (d.origin_record_id === view.merge.incoming_id ? '#ff9900' : '#00bb66')
      : view.colorOf.get(d.class_name) || '#1a73e8';
    const pct = (v, total) => `${(v / total) * 100}%`;

    if (view.shape === 'fill') {
      // 实心：完全不透明填充 + 实线边框（透明度 0，alpha=1），填充色用 --c-soft
      const node = document.createElement('div');
      node.className = 'bbox fill';
      node.dataset.i = String(i);
      node.style.setProperty('--c', color);
      node.style.setProperty(
        '--c-soft',
        view.merge || view.fusion ? color : classColorA(Math.max(0, view.classes.indexOf(d.class_name)), 1)
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

  if (view.pose && view.showSkeleton) {
    wrap.appendChild(createPoseLayer(shown.map(i => view.dets[i]), view.pose, view.naturalW, view.naturalH));
  }

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
  if (view.dets.some(d => d.behavior?.roi?.status === 'unavailable')) notes.push('物体检测不可用，保留低头');

  const unit = view.shape === 'dot' ? '个点' : '个框';
  view.q('[data-role="count"]').textContent =
    (view.headCounting
      ? `人数（当前筛选）${shown.length} 人 / 模型检出 ${view.dets.length} 个人头`
      : `图上 ${shown.length} ${unit} / 共检出 ${view.dets.length} 个`) +
    (notes.length ? `（${notes.join('；')}）` : '') +
    (view.dets.some(d => d.behavior) ? '｜' + (view.pose?.exclude_unknown ? ['举手', '抬头', '低头', '疑似看手机', '疑似阅读'] : ['举手', '抬头', '低头', '疑似看手机', '疑似阅读', '无法判断']).map(
      state => `${state} ${shown.filter(i => view.dets[i].behavior?.state === state).length} 人`
    ).join(' · ') : '');
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


export { renderDetectResult, classColor, classColorA, createPoseLayer };
