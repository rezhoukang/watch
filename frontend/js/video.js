import { el, esc, api, banner, showError } from './core.js?v=72';
import { readDetectOptions } from './models.js?v=72';
import { pollJob } from './jobs.js?v=72';
import { showHeadChart } from './charts.js?v=72';
import { loadOutputs } from './outputs.js?v=72';

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
    const headDetection = !!el('vidHeadDetection').checked;
    const conf = headDetection ? '0.45' : el('vidConf').value || '0.38';
    const stride = el('vidStride').value || '1';
    const render = el('vidRender').value || 'box';
    const { start, end } = readVideoRange(item);
    const { weights, isWorld, classes, isRtdetr, isPose, rtdetrBase, mergeIou, mergeConfidence, headDownRThreshold, headUpRThreshold, poseRoi, roiConfidence } = readDetectOptions('vid');
    const rangeText = end == null ? `全片` : `${fmtDuration(start)} ~ ${fmtDuration(end)}`;
    const renderText = render === 'dot' ? '中心点' : '框';
    const modeText = headDetection ? `目标检测｜${rtdetrBase ? 'RT-DETRv4 0.25 基准 + medium 0.45 补齐' : 'medium 0.45 基准 + RT-DETRv4 0.25 补齐'}｜640｜IoU ≤ ${mergeIou} 补齐` : isWorld
      ? `YOLO-World${classes ? `｜prompt「${classes}」` : '｜prompt=权重内置'}`
      : isRtdetr ? 'RT-DETRv4 人头检测｜640' : isPose ? 'YOLO Pose 人体骨架' : '普通 YOLO';

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
    if (isPose) { q.set('head_down_r_threshold', String(headDownRThreshold)); q.set('head_up_r_threshold', String(headUpRThreshold)); q.set('pose_roi', String(poseRoi)); q.set('roi_confidence', String(roiConfidence)); }
    if (headDetection) q.set('head_detection', 'true');
    if (headDetection) q.set('merge_iou', String(mergeIou));
    if (headDetection) q.set('merge_confidence', String(mergeConfidence));
    if (rtdetrBase) q.set('rtdetr_base', 'true');
    if (classes) q.set('classes', classes);
    const res = await api(
      `/api/detect/video/${encodeURIComponent(filename)}?${q}`,
      { method: 'POST' },
      30000
    );
    if (!res || !res.job_id) {
      throw new Error(`后端未返回 job_id：${JSON.stringify(res).slice(0, 200)}`);
    }
    if (headDetection && !res.head_detection) throw new Error('目标检测未生效，请重启服务后刷新页面重试。');
    if (rtdetrBase && !res.rtdetr_base) throw new Error('RT-DETR 基准未生效，请重启服务后刷新页面。');
    if (headDetection && Number(res.merge_iou ?? 0) !== mergeIou) throw new Error('合并 IoU 阈值未生效，请重启服务后刷新页面。');
    if (headDetection && Number(res.merge_confidence ?? 0) !== mergeConfidence) throw new Error('合并置信度门槛未生效，请重启服务后刷新页面。');

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
          const rtConf = job.parameters?.[job.parameters?.rtdetr_base ? 'base' : 'incoming']?.confidence ?? 0.2;
          box.innerHTML = (job.head_detection ? `<p>目标检测融合结果：${job.parameters?.rtdetr_base ? `RT-DETRv4 ${rtConf} 基准 + medium 0.45 补齐` : `medium 0.45 基准 + RT-DETRv4 ${rtConf} 补齐`}，640；合并 IoU ≤ ${job.parameters?.merge_iou ?? 0}；绿框为基准，橙框为补充。</p>` : '') +
            `<video src="${esc(job.result_url)}" controls autoplay></video>`;
          if (job.head_detection) {
            const gate = document.createElement('p');
            gate.textContent = `自定义 IoU 仅用于待补入框置信度 > ${job.parameters?.merge_confidence ?? 0}；其余仍要求 IoU=0。`;
            box.appendChild(gate);
          }
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


export { loadVideos };
