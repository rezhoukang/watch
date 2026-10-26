import { el, esc, api } from './core.js?v=94';
import { bindImageHover } from './preview.js?v=94';

// Independent module: anonymous Head IDs, original-image crops, local VLM only.
let images = [], record = null, activeCrop = '', activeJob = null, timer = null;
let loading = false, modelReady = false, pointerStart = null, polling = false;
const selectedImages = new Set(), selectedCrops = new Set();
const terminal = new Set(['completed', 'failed', 'cancelled']);
const json = (data, method = 'POST') => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
const error = err => { el('actError').textContent = err.message || String(err); };
const clearError = () => { el('actError').textContent = ''; };
const config = () => ({ model: el('actModel').value || '4b', quantization: el('actQuantization').value || '4bit' });
const expansion = () => ({ side: Number(el('actSide').value), top: Number(el('actTop').value), below: Number(el('actBelow').value) });
const busy = () => loading || !!activeJob;
const current = () => record?.crops.find(item => item.crop_id === activeCrop);
const sourceInfo = () => record?.images.find(item => item.image_index === current()?.image_index);
const fileUrl = (file, revision = 0) => `/api/actions/runs/${record.run_id}/files/${file}?v=${revision}`;

function sync() {
  const running = busy(), count = record?.crops.length || 0;
  el('actSelectionCount').textContent = `已选 ${selectedImages.size} 张`;
  el('actSelectAll').checked = images.length > 0 && selectedImages.size === images.length;
  el('actPrepare').disabled = running || !selectedImages.size;
  el('actReexpand').disabled = running || !count;
  el('actAnalyze').disabled = running || !count || !modelReady;
  el('actAnalyze').textContent = selectedCrops.size ? `分析选中 ${selectedCrops.size} 个目标` : '分析全部裁剪';
  el('actRetry').disabled = running || !modelReady || !record?.crops.some(item => item.status === 'failed');
  el('actCancel').hidden = !activeJob || activeJob.kind === 'download';
  el('actSaveRegion').disabled = running || !current();
  el('actCropSelectAll').checked = count > 0 && selectedCrops.size === count;
  for (const id of ['actModel', 'actQuantization', 'actDownload', 'actUnload', 'actRun', 'actSide', 'actTop',
    'actBelow', 'actSource', 'actX1', 'actY1', 'actX2', 'actY2', 'actUpload', 'actRefresh', 'actSelectAll',
    'actCropSelectAll', 'actDraw', 'actResetRegion', 'actDefaultExpansion']) el(id).disabled = running;
  el('actImageGrid').querySelectorAll('input').forEach(input => { input.disabled = running; });
  el('actCropGrid').querySelectorAll('input, button').forEach(input => { input.disabled = running; });
}

function expansionLabels() {
  for (const [id, unit] of [['Side', '头宽'], ['Top', '头高'], ['Below', '头高']]) {
    el(`act${id}Value`).textContent = `${Number(el(`act${id}`).value).toFixed(2)} 倍${unit}`;
  }
}

async function checkModel() {
  const options = config();
  const state = await api(`/api/actions/status?${new URLSearchParams(options)}`);
  modelReady = state.available;
  el('actModelStatus').textContent = state.loaded ? '模型已加载 · 本地推理' : state.message;
  el('actModelStatus').style.color = state.available ? '#187856' : '#986627';
  if (state.active_job && !activeJob) watch(state.active_job);
  sync();
  return state;
}

function renderImages() {
  const grid = el('actImageGrid'); grid.innerHTML = '';
  for (const image of images) {
    const card = document.createElement('figure');
    card.className = selectedImages.has(image.filename) ? 'is-selected' : '';
    card.innerHTML = `<img src="${esc(image.url)}" alt="${esc(image.filename)}" loading="lazy"><figcaption title="${esc(image.filename)}">${esc(image.filename)}</figcaption>`;
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox';
    checkbox.checked = selectedImages.has(image.filename);
    checkbox.setAttribute('aria-label', `选择动作图片 ${image.filename}`);
    checkbox.addEventListener('click', event => event.stopPropagation());
    const select = () => {
      if (checkbox.checked) selectedImages.add(image.filename); else selectedImages.delete(image.filename);
      card.classList.toggle('is-selected', checkbox.checked); sync();
    };
    checkbox.addEventListener('change', select);
    card.addEventListener('click', () => { if (!busy()) { checkbox.checked = !checkbox.checked; select(); } });
    card.appendChild(checkbox); grid.appendChild(card);
    bindImageHover(card.querySelector('img'), () => ({ src: image.url, name: image.filename }));
  }
  if (!images.length) grid.innerHTML = '<div class="empty">上传课堂图片后开始。</div>';
  sync();
}

async function loadImages() {
  const data = await api('/api/images'); images = data.images;
  for (const name of selectedImages) if (!images.some(image => image.filename === name)) selectedImages.delete(name);
  renderImages();
}

async function loadRuns() {
  const data = await api('/api/actions/runs');
  el('actRun').innerHTML = '<option value="">选择历史批次</option>' + data.runs.map(run =>
    `<option value="${esc(run.run_id)}">${esc(new Date(run.created_at).toLocaleString())} · ${run.crop_count} 个目标 · ${esc(run.filenames[0] || '')}</option>`).join('');
  if (record) el('actRun').value = record.run_id;
  return data.runs;
}

function applyRecord(data, keepSelection = true) {
  const changed = record?.run_id !== data.run_id;
  record = data;
  if (changed || !keepSelection) selectedCrops.clear();
  for (const id of selectedCrops) if (!record.crops.some(item => item.crop_id === id)) selectedCrops.delete(id);
  if (!record.crops.some(item => item.crop_id === activeCrop)) activeCrop = record.crops[0]?.crop_id || '';
  if (changed) {
    for (const [key, id] of [['side', 'actSide'], ['top', 'actTop'], ['below', 'actBelow']]) el(id).value = record.expansion[key];
    expansionLabels();
  }
  const runErrors = record.errors || [];
  el('actErrors').hidden = !runErrors.length;
  el('actErrorList').innerHTML = runErrors.map(item => `<p>${esc(item.source)}：${esc(item.message)}</p>`).join('');
  el('actRunSummary').textContent = `${record.images.length} 张图片 · ${record.crops.length} 个目标`;
  renderCrops(); renderEditor(); sync();
}

async function openRun(id) {
  if (!id) return;
  applyRecord(await api(`/api/actions/runs/${id}`));
  el('actRun').value = id;
}

function renderCrops() {
  const grid = el('actCropGrid'); grid.innerHTML = '';
  const items = record?.crops || [], filter = el('actFilter').value || 'all';
  const counts = {};
  for (const item of items) {
    const state = item.action?.state || (item.status === 'failed' ? '分析失败' : '待分析');
    counts[state] = (counts[state] || 0)+1;
    if (filter !== 'all' && filter !== item.status && filter !== item.action?.state) continue;
    const card = document.createElement('article');
    card.className = `action-crop-card${item.crop_id === activeCrop ? ' is-active' : ''}`;
    card.innerHTML = `<div class="action-crop-title"><b>${esc(item.crop_id)}</b><input type="checkbox" aria-label="选择目标 ${esc(item.crop_id)}"></div>
      <img src="${esc(fileUrl(item.marked_file, item.revision))}" alt="${esc(item.crop_id)} 目标裁剪" loading="lazy">
      <div class="action-crop-info"><span class="action-state${item.status === 'failed' ? ' is-failed' : state === '无法判断' ? ' is-uncertain' : ''}">${esc(state)}</span>
      <p class="action-evidence">${esc(item.action?.evidence || item.error || '检查范围后启动 VLM 分析')}</p>
      ${item.action?.uncertainty ? `<p class="action-reason">${esc(item.action.uncertainty)}</p>` : ''}
      <p class="action-crop-source" title="${esc(item.source)}">${esc(item.source)}${item.elapsed_seconds != null ? ` · ${item.elapsed_seconds}s` : ''}</p>
      <button class="btn small ghost" type="button">检查 / 调整区域</button></div>`;
    const checkbox = card.querySelector('input'); checkbox.checked = selectedCrops.has(item.crop_id);
    checkbox.addEventListener('change', () => { if (checkbox.checked) selectedCrops.add(item.crop_id); else selectedCrops.delete(item.crop_id); sync(); });
    const select = () => { activeCrop = item.crop_id; renderCrops(); renderEditor(); sync(); };
    card.querySelector('button').addEventListener('click', select); card.querySelector('img').addEventListener('click', select);
    bindImageHover(card.querySelector('img'), () => ({ src: fileUrl(item.marked_file, item.revision), name: `${item.crop_id} · ${state}`, enlarge: true }));
    grid.appendChild(card);
  }
  if (!grid.children.length) grid.innerHTML = `<div class="empty">${items.length ? '当前筛选下没有目标。' : '暂无裁剪，请先选择图片进行 Head 检测。'}</div>`;
  el('actCounts').textContent = Object.entries(counts).map(([name, count]) => `${name} ${count}`).join(' · ') || '暂无目标';
}

function editorBox() { return ['actX1', 'actY1', 'actX2', 'actY2'].map(id => Number(el(id).value)); }
function setBox(box) {
  ['actX1', 'actY1', 'actX2', 'actY2'].forEach((id, index) => { el(id).value = Math.round(box[index]); });
  drawOverlay(box);
}
function drawOverlay(box = editorBox()) {
  const item = current(), info = sourceInfo();
  if (!item || !info) return;
  const rect = value => `left:${value[0]/info.width*100}%;top:${value[1]/info.height*100}%;width:${(value[2]-value[0])/info.width*100}%;height:${(value[3]-value[1])/info.height*100}%`;
  el('actRegionOverlay').innerHTML = `<div class="action-region" style="${rect(box)}"><span class="action-region-label">${esc(activeCrop)}</span></div><div class="action-head" style="${rect(item.head_bbox)}"></div>`;
}
function renderEditor() {
  const item = current(); el('actEditor').hidden = !item; el('actPreviewEmpty').hidden = !!item;
  if (!item) return;
  el('actSource').innerHTML = record.images.filter(info => record.crops.some(crop => crop.image_index === info.image_index))
    .map(info => `<option value="${info.image_index}">${esc(info.source)}</option>`).join('');
  el('actSource').value = item.image_index;
  el('actSourceImg').src = fileUrl(item.source_file);
  el('actTargetName').textContent = `${item.crop_id} · 原图像素坐标`;
  setBox(item.region_bbox);
}
function previewExpansion() {
  expansionLabels(); const item = current(), info = sourceInfo(); if (!item || !info) return;
  const [x1, y1, x2, y2] = item.head_bbox, w = x2-x1, h = y2-y1, options = expansion();
  setBox([Math.max(0, Math.floor(x1-w*options.side)), Math.max(0, Math.floor(y1-h*options.top)),
    Math.min(info.width, Math.ceil(x2+w*options.side)), Math.min(info.height, Math.ceil(y2+h*options.below))]);
}

async function poll() {
  if (!activeJob || polling) return;
  polling = true;
  try {
    const job = await api(`/api/actions/jobs/${activeJob.job_id}`);
    el('actProgress').innerHTML = `${esc(job.message)} · ${job.progress}/${job.total}${job.errors ? ` · ${job.errors} 项失败` : ''}${job.cached ? ` · 复用 ${job.cached} 项` : ''}<progress value="${job.progress}" max="${Math.max(1, job.total)}"></progress>`;
    if (job.run_id) await openRun(job.run_id);
    if (terminal.has(job.status)) {
      clearInterval(timer); timer = null; activeJob = null;
      await loadRuns(); await checkModel();
      if (job.status === 'failed') error(new Error(job.message));
    }
  } catch (err) {
    clearInterval(timer); timer = null; activeJob = null; error(err);
  }
  polling = false; sync();
}
function watch(job) {
  activeJob = job; clearInterval(timer); timer = setInterval(poll, 1800); sync();
  poll();
}
async function command(work) {
  if (busy()) return;
  loading = true; clearError(); sync();
  try { await work(); } catch (err) { error(err); }
  finally { loading = false; sync(); }
}

el('actPrepare').addEventListener('click', () => command(async () => {
  const job = await api('/api/actions/runs', json({ filenames: [...selectedImages], ...expansion() }));
  watch(job);
}));
el('actReexpand').addEventListener('click', () => command(async () => {
  applyRecord(await api(`/api/actions/runs/${record.run_id}/expansion`, json(expansion())));
}));
el('actSaveRegion').addEventListener('click', () => command(async () => {
  applyRecord(await api(`/api/actions/runs/${record.run_id}/crops/${activeCrop}`, json({ bbox: editorBox() }, 'PATCH')));
}));
el('actResetRegion').addEventListener('click', renderEditor);
el('actDefaultExpansion').addEventListener('click', () => {
  el('actSide').value = 1; el('actTop').value = 0; el('actBelow').value = 1.3;
  previewExpansion();
});
async function startAnalysis(ids) {
  watch(await api(`/api/actions/runs/${record.run_id}/analyze`, json({ ...config(), crop_ids: ids, force: !!el('actForce').checked })));
}
el('actAnalyze').addEventListener('click', () => command(() => startAnalysis([...selectedCrops])));
el('actRetry').addEventListener('click', () => command(() => startAnalysis(record.crops.filter(item => item.status === 'failed').map(item => item.crop_id))));
el('actCancel').addEventListener('click', async () => {
  try { await api(`/api/actions/jobs/${activeJob.job_id}/cancel`, json({})); await poll(); } catch (err) { error(err); }
});
el('actCheckModel').addEventListener('click', () => checkModel().catch(error));
for (const id of ['actModel', 'actQuantization']) el(id).addEventListener('change', () => checkModel().catch(error));
el('actDownload').addEventListener('click', () => command(async () => { watch(await api('/api/actions/model/download', json(config()))); }));
el('actUnload').addEventListener('click', () => command(async () => { await api('/api/actions/model/unload', json({})); await checkModel(); }));
el('actRefresh').addEventListener('click', () => command(loadImages));
el('actRun').addEventListener('change', () => command(() => openRun(el('actRun').value)));
el('actSource').addEventListener('change', () => {
  const item = record.crops.find(crop => crop.image_index === Number(el('actSource').value));
  if (item) { activeCrop = item.crop_id; renderCrops(); renderEditor(); }
});
el('actSelectAll').addEventListener('change', () => {
  selectedImages.clear(); if (el('actSelectAll').checked) images.forEach(image => selectedImages.add(image.filename)); renderImages();
});
el('actCropSelectAll').addEventListener('change', () => {
  selectedCrops.clear(); if (el('actCropSelectAll').checked) record?.crops.forEach(item => selectedCrops.add(item.crop_id)); renderCrops(); sync();
});
el('actFilter').addEventListener('change', renderCrops);
for (const id of ['actSide', 'actTop', 'actBelow']) el(id).addEventListener('input', previewExpansion);
for (const id of ['actX1', 'actY1', 'actX2', 'actY2']) el(id).addEventListener('input', () => drawOverlay());
el('actUpload').addEventListener('click', () => el('actUploadInput').click());
el('actUploadInput').addEventListener('change', () => command(async () => {
  const input = el('actUploadInput'), failures = [];
  for (const file of input.files) {
    const form = new FormData(); form.append('file', file);
    try { const saved = await api('/api/upload', { method: 'POST', body: form }); selectedImages.add(saved.filename); }
    catch (err) { failures.push(`${file.name}：${err.message}`); }
  }
  input.value = ''; await loadImages();
  if (failures.length) throw new Error(failures.join('；'));
}));

el('actDraw').addEventListener('change', () => el('actSourceView').classList.toggle('is-drawing', el('actDraw').checked));
function pointerPosition(event) {
  const bounds = el('actSourceImg').getBoundingClientRect(), info = sourceInfo();
  return [Math.round(Math.max(0, Math.min(1, (event.clientX-bounds.left)/bounds.width))*info.width),
    Math.round(Math.max(0, Math.min(1, (event.clientY-bounds.top)/bounds.height))*info.height)];
}
el('actSourceView').addEventListener('pointerdown', event => {
  if (!el('actDraw').checked || busy() || !current() || event.button !== 0) return;
  event.preventDefault(); pointerStart = pointerPosition(event); el('actSourceView').setPointerCapture(event.pointerId);
});
el('actSourceView').addEventListener('pointermove', event => {
  if (!pointerStart) return;
  const end = pointerPosition(event);
  setBox([Math.min(pointerStart[0], end[0]), Math.min(pointerStart[1], end[1]), Math.max(pointerStart[0], end[0]), Math.max(pointerStart[1], end[1])]);
});
el('actSourceView').addEventListener('pointerup', () => { pointerStart = null; });
el('actSourceView').addEventListener('pointercancel', () => { pointerStart = null; });

async function loadActions() {
  try {
    clearError();
    const [, runs] = await Promise.all([loadImages(), loadRuns(), checkModel()]);
    if (!record && runs.length && !activeJob) await openRun(runs[0].run_id);
    expansionLabels(); sync();
  } catch (err) { error(err); }
}

export { loadActions };
