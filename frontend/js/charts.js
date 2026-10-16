import { el, esc, api } from './core.js?v=89';

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
  const video = el('videoResult').querySelector('video');

  const wrap = document.createElement('div');
  wrap.className = 'headchart';
  wrap.innerHTML =
    `<div class="headchart-head">` +
    `<span class="headchart-hint">⇔ 滚动/拖动平移 ｜ Ctrl+滚轮 缩放宽度</span></div>` +
    `<div class="headchart-stats" aria-live="polite"></div>` +
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
  let dragging = false, dragMoved = false, dragStartX = 0, dragStartScroll = 0;
  scroll.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    dragging = true;
    dragMoved = false;
    dragStartX = e.clientX;
    dragStartScroll = scroll.scrollLeft;
    scroll.classList.add('dragging');
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    if (Math.abs(e.clientX - dragStartX) > 4) dragMoved = true;
    scroll.scrollLeft = dragStartScroll - (e.clientX - dragStartX);
  });
  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    scroll.classList.remove('dragging');
  });

  // 帧号使用原视频时间轴；推理结果视频从 start_frame 开始。
  scroll.addEventListener('click', (e) => {
    if (dragMoved || !video || !view?.all.length) return;
    const frame = xToFrame(e.clientX - scroll.getBoundingClientRect().left + scroll.scrollLeft);
    const nearest = view.all.reduce((best, point) =>
      Math.abs(point[0] - frame) < Math.abs(best[0] - frame) ? point : best
    );
    const seek = () => {
      const seconds = Math.max(0, (nearest[0] - view.startFrame) / view.fps);
      video.currentTime = Number.isFinite(video.duration) ? Math.min(seconds, video.duration) : seconds;
    };
    if (video.readyState >= 1) seek();
    else video.addEventListener('loadedmetadata', seek, { once: true });
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
        startFrame: Number(d?.start_frame) || 0,
        firstFrame: raw.length ? raw[0][0] : 0,
        lastFrame: raw.length ? raw[raw.length - 1][0] : 0,
      };
      // 统计全部原始采样数据，不使用用于绘图的降采样点。
      const stats = wrap.querySelector('.headchart-stats');
      if (raw.length) {
        const totals = raw.reduce((acc, [, count]) => ({
          min: Math.min(acc.min, count), max: Math.max(acc.max, count), sum: acc.sum + count,
        }), { min: Infinity, max: -Infinity, sum: 0 });
        stats.textContent = `最高人数：${totals.max} 人 ｜ 最低人数：${totals.min} 人 ｜ 全局平均人数：${(totals.sum / raw.length).toFixed(2)} 人`;
        stats.title = '按本次推理时间段的全部采样帧计算；抽帧间隔大于 1 时使用实际推理帧。';
      }
      // 标题补充总帧数
      const hint = wrap.querySelector('.headchart-hint');
      if (hint && raw.length) {
        hint.textContent = `共 ${raw.length} 个采样帧 ｜ 点击跳转视频 ｜ ⇔ 滚动/拖动平移 ｜ Ctrl+滚轮 缩放`;
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


export { showHeadChart };
