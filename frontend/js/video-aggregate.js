import { el, esc, api } from './core.js?v=95';

const base = '/api/video-aggregate';
const terminal = new Set(['completed', 'failed', 'cancelled']);
let job = null, record = null, activeFrame = null, timer = null, loading = false, polling = false;
const post = data => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
const timeLabel = seconds => `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(2).padStart(5, '0')}`;
const fileUrl = filename => `${base}/runs/${record.run_id}/files/${filename}`;
const fail = error => { el('vaggError').textContent = error.message || String(error); };

function sync() {
  const busy = loading || !!job;
  for (const id of ['vaggUpload', 'vaggVideo', 'vaggRefresh', 'vaggInterval']) el(id).disabled = busy;
  el('vaggStart').disabled = busy || !el('vaggVideo').value;
  el('vaggCancel').hidden = !job;
}

function player(url) {
  const video = el('vaggPlayer');
  if (video.getAttribute('src') !== url) video.src = url;
  video.hidden = false;
}

async function metadata() {
  const name = el('vaggVideo').value;
  if (!name) return;
  const info = await api(`/api/videos/${encodeURIComponent(name)}/info`);
  const interval = Number(el('vaggInterval').value);
  const seconds = info.duration ?? info.duration_seconds ?? 0;
  el('vaggMeta').textContent = `视频 ${timeLabel(seconds)} · ${info.fps} FPS · 每 ${interval} 秒采样一次`;
  if (!job) player(`/videos/${encodeURIComponent(name)}`);
}

async function lists() {
  const videos = await api('/api/videos');
  const selected = el('vaggVideo').value;
  const options = videos.videos.filter(video => video.filename.toLowerCase().endsWith('.mp4'));
  el('vaggVideo').innerHTML = '<option value="">选择 MP4</option>' + options.map(video =>
    `<option value="${esc(video.filename)}">${esc(video.filename)}</option>`).join('');
  if (options.some(video => video.filename === selected)) el('vaggVideo').value = selected;
  sync();
}

function renderFrames() {
  const grid = el('vaggFrames'), scroll = grid.scrollTop;
  grid.innerHTML = record.video.frames.map(frame => {
    const counts = Object.entries(frame.counts).map(([state, count]) => `${state} ${count}`).join(' · ');
    return `<button class="video-aggregate-card${frame.image_index === activeFrame ? ' is-active' : ''}" data-frame="${frame.image_index}">
      <img src="${esc(fileUrl(frame.result_file || frame.source_file))}" loading="lazy" alt="${esc(timeLabel(frame.time_seconds))} 采样帧">
      <strong>${esc(timeLabel(frame.time_seconds))}</strong><small>${frame.detected ? `${frame.count} 个目标` : '待目标检测'}${frame.provisional ? ' · 初步结果' : ''}</small>
      <small>${esc(counts || '等待目标检测')}</small></button>`;
  }).join('') || '<p class="empty">等待 OpenCV 提取采样帧。</p>';
  grid.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
    activeFrame = Number(button.dataset.frame);
    renderFrames();
    showFrame(true).catch(fail);
  }));
  grid.scrollTop = scroll;
}

async function showFrame(seek = false) {
  if (!record || activeFrame == null) return;
  const runId = record.run_id, frameIndex = activeFrame;
  const data = await api(`${base}/runs/${runId}/frames/${frameIndex}`);
  if (record?.run_id !== runId || activeFrame !== frameIndex) return;
  el('vaggDetail').hidden = false;
  el('vaggFrameTitle').textContent = `${timeLabel(data.time_seconds)} · 第 ${data.frame_index} 帧 · ${data.targets.length} 个目标`;
  el('vaggFrameImage').src = fileUrl(data.result_file || data.source_file);
  el('vaggTargets').innerHTML = data.targets.map(target => {
    const action = target.action;
    const stage = target.status === 'failed' ? '分析失败' : action?.provisional ? 'Head 初步结果' : action?.fallback ? '规则兜底' : action ? '两阶段完成' : '待分析';
    return `<tr><td>${esc(target.crop_id)}</td><td>${esc(action?.state || stage)}</td><td>${esc(target.error || action?.evidence || '')}</td><td>${esc(stage)}</td></tr>`;
  }).join('') || '<tr><td colspan="4">该帧尚未检出目标。</td></tr>';
  if (seek) {
    const video = el('vaggPlayer');
    if (video.readyState) video.currentTime = data.time_seconds;
    else video.addEventListener('loadedmetadata', () => { video.currentTime = data.time_seconds; }, { once: true });
  }
}

async function openRun(id) {
  const changed = record?.run_id !== id;
  record = await api(`${base}/runs/${id}`);
  if (changed) {
    activeFrame = null;
    el('vaggDetail').hidden = true;
    player(record.video_url);
  }
  el('vaggExport').hidden = false;
  el('vaggExport').href = `${base}/runs/${id}/export`;
  const names = { completed: '完成', failed: '失败', cancelled: '已取消', interrupted: '已中断', extracting: '抽帧中', detecting: '目标检测中', analyzing: '动作分析中', rendering: '生成结果图' };
  el('vaggSummary').textContent = `${record.source} · ${record.video.frames.length} 个采样帧 · ${record.crop_count} 个目标 · ${names[record.status] || record.status}${record.video.actual_interval_seconds ? ` · 实际间隔 ${record.video.actual_interval_seconds.toFixed(3)}s` : ''}`;
  if (record.errors.length) el('vaggError').textContent = record.errors.map(item => item.message).join('；');
  renderFrames();
  if (activeFrame != null) await showFrame();
}

async function poll() {
  if (!job || polling) return;
  polling = true;
  try {
    job = await api(`${base}/jobs/${job.job_id}`);
    const bounded = ['head', 'context'].includes(job.stage) && job.total > 0;
    el('vaggProgress').innerHTML = `${esc(job.message)}${job.errors ? ` · ${job.errors} 项分析失败` : ''}<progress ${bounded ? `max="${job.total}" value="${job.progress}"` : ''}></progress>`;
    await openRun(job.run_id);
    if (terminal.has(job.status)) {
      job = null;
      clearInterval(timer);
      timer = null;
      await lists();
    }
  } catch (error) {
    fail(error);
    clearInterval(timer);
    timer = null;
    job = null;
  } finally {
    polling = false;
    sync();
  }
}

function watch(value) {
  job = value;
  clearInterval(timer);
  timer = setInterval(poll, 1500);
  sync();
  poll();
}

async function operation(work) {
  if (loading || job) return;
  loading = true;
  el('vaggError').textContent = '';
  sync();
  try { await work(); } catch (error) { fail(error); }
  finally { loading = false; sync(); }
}

el('vaggUpload').addEventListener('click', () => el('vaggUploadInput').click());
el('vaggUploadInput').addEventListener('change', () => operation(async () => {
  const input = el('vaggUploadInput'), file = input.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.mp4')) throw new Error('请选择 MP4 视频');
  const form = new FormData(); form.append('file', file);
  try {
    const result = await api('/api/upload_video', { method: 'POST', body: form }, 600000);
    await lists();
    el('vaggVideo').value = result.filename;
    await metadata();
  } finally { input.value = ''; }
}));
el('vaggRefresh').addEventListener('click', () => operation(lists));
el('vaggVideo').addEventListener('change', () => metadata().catch(fail));
el('vaggInterval').addEventListener('change', () => metadata().catch(fail));
el('vaggStart').addEventListener('click', () => operation(async () => {
  const interval = Number(el('vaggInterval').value);
  if (!Number.isFinite(interval) || interval < 0.1 || interval > 86400) throw new Error('抽帧间隔需在 0.1～86400 秒之间');
  watch(await api(`${base}/jobs`, post({ filename: el('vaggVideo').value, interval_seconds: interval })));
}));
el('vaggCancel').addEventListener('click', async () => {
  if (!job) return;
  try { await api(`${base}/jobs/${job.job_id}/cancel`, post({})); await poll(); }
  catch (error) { fail(error); }
});

async function loadVideoAggregate() {
  try {
    await lists();
    const data = await api(`${base}/jobs/active`);
    if (data.job && !job) watch(data.job);
  } catch (error) { fail(error); }
}

sync();
export { loadVideoAggregate };
