import { el, esc, api, showError } from './core.js?v=71';

const selected = new Set();
let filenames = [], current = null, activeJob = null, page = 0, loadVersion = 0;
const PAGE_SIZE = 60;
const fileUrl = (run, file) => `/api/head-cuts/${run}/files/${file.split('/').map(encodeURIComponent).join('/')}`;
const running = status => status === 'running' || status === 'queued';

function syncSelection() {
  const all = el('cutSelectAll');
  all.checked = filenames.length > 0 && selected.size === filenames.length;
  all.indeterminate = selected.size > 0 && selected.size < filenames.length;
  el('cutSelectionCount').textContent = `已选 ${selected.size} 张`;
  el('cutInfer').disabled = !!activeJob || !selected.size;
  el('cutCancel').hidden = !activeJob;
  for (const id of ['cutRtdetrBase', 'cutMergeIou', 'cutMergeConfidence']) el(id).disabled = !!activeJob;
  el('cutImageGrid').querySelectorAll('[data-file]').forEach(input => { input.checked = selected.has(input.dataset.file); });
}

function renderImages(images) {
  filenames = images.map(i => i.filename);
  for (const name of selected) if (!filenames.includes(name)) selected.delete(name);
  const grid = el('cutImageGrid');
  grid.innerHTML = '';
  for (const image of images) {
    const card = document.createElement('figure');
    card.innerHTML = `<img src="${esc(image.url)}" alt="${esc(image.filename)}" loading="lazy"><figcaption>${esc(image.filename)}</figcaption>`;
    const label = document.createElement('label'); label.className = 'head-cut-select';
    label.addEventListener('click', event => event.stopPropagation());
    const check = document.createElement('input'); check.type = 'checkbox'; check.dataset.file = image.filename;
    check.setAttribute('aria-label', `选择 ${image.filename}`);
    check.addEventListener('change', () => { if (check.checked) selected.add(image.filename); else selected.delete(image.filename); syncSelection(); });
    label.appendChild(check); card.appendChild(label);
    card.addEventListener('click', event => { if (event.target === check) return; check.checked = !selected.has(image.filename); if (check.checked) selected.add(image.filename); else selected.delete(image.filename); syncSelection(); });
    grid.appendChild(card);
  }
  syncSelection();
}

async function loadBatches(preferred = current?.run_id) {
  const data = await api('/api/head-cuts');
  const select = el('cutBatchSelect');
  select.innerHTML = '';
  for (const run of data.runs) {
    const option = document.createElement('option'); option.value = run.run_id;
    option.textContent = `${new Date(run.created_at).toLocaleString()} · ${run.crop_count} 个 · ${run.run_id.slice(0, 8)}`;
    select.appendChild(option);
  }
  const target = data.runs.some(r => r.run_id === preferred) ? preferred : data.runs[0]?.run_id;
  if (!target) { select.innerHTML = '<option value="">暂无批次</option>'; return; }
  select.value = target;
  await openBatch(target);
}

async function openBatch(runId) {
  const version = ++loadVersion;
  const record = await api(`/api/head-cuts/${runId}`);
  if (version !== loadVersion) return;
  current = record; page = 0;
  const oldSource = el('cutSourceFilter').value;
  el('cutSourceFilter').innerHTML = '<option value="">全部图片</option>';
  for (const item of record.inputs) {
    const option = document.createElement('option'); option.value = item.filename; option.textContent = item.filename;
    el('cutSourceFilter').appendChild(option);
  }
  el('cutSourceFilter').value = record.inputs.some(i => i.filename === oldSource) ? oldSource : '';
  renderHeads();
  if (running(record.status)) {
    if (!activeJob) activeJob = runId;
    syncSelection();
    setTimeout(() => pollBatch(runId, version), 1200);
  } else if (activeJob === runId) {
    activeJob = null; syncSelection();
  }
}

async function pollBatch(runId, version) {
  try {
    const record = await api(`/api/head-cuts/${runId}`);
    if (version !== loadVersion && (current?.run_id === runId || activeJob !== runId)) return;
    if (version === loadVersion) { current = record; renderHeads(); }
    if (running(record.status)) {
      setTimeout(() => pollBatch(runId, version), 1200);
    } else {
      if (activeJob === runId) activeJob = null;
      syncSelection();
      if (version === loadVersion) el('cutProgress').textContent = record.message;
    }
  } catch (error) {
    if (activeJob === runId) activeJob = null;
    syncSelection(); showError(el('cutProgress'), error, '读取头切割进度失败');
  }
}

function renderHeads() {
  if (!current) return;
  const record = current;
  el('cutProgress').textContent = `${record.progress}/${record.total} · ${record.message}`;
  el('cutDownload').hidden = running(record.status);
  el('cutDownload').href = `/api/head-cuts/${record.run_id}/download`;
  const source = el('cutSourceFilter').value;
  const search = el('cutSearch').value.trim().toLowerCase();
  const filtered = record.crops.filter(c => (!source || c.source === source) && `${c.crop_id} ${c.class_name} ${c.mark || ''}`.toLowerCase().includes(search));
  const pages = Math.max(1, Math.ceil(filtered.length/PAGE_SIZE));
  page = Math.min(page, pages-1);
  el('cutHeadCount').textContent = `共 ${record.crop_count} 个裁剪 · 当前筛选 ${filtered.length} 个 · 已处理 ${record.progress}/${record.total} 张`;
  const previews = el('cutSourcePreviews'); previews.innerHTML = '';
  for (const image of record.images.filter(i => !source || i.source === source)) {
    const link = document.createElement('a'); link.href = fileUrl(record.run_id, image.annotated_file); link.target = '_blank';
    link.innerHTML = `<img src="${esc(link.href)}" alt="${esc(image.source)}编号总览" loading="lazy"><span>${esc(image.source)} · ${image.crop_count} 个</span>`;
    previews.appendChild(link);
  }
  const grid = el('cutHeadGrid');
  if (document.activeElement && grid.contains(document.activeElement)) return;
  grid.innerHTML = '';
  for (const crop of filtered.slice(page*PAGE_SIZE, (page+1)*PAGE_SIZE)) {
    const figure = document.createElement('figure');
    figure.innerHTML = `<a href="${esc(fileUrl(record.run_id, crop.raw_file))}" target="_blank"><img src="${esc(fileUrl(record.run_id, crop.marked_file))}" alt="${esc(crop.crop_id)}" loading="lazy"></a><figcaption><strong>${esc(crop.crop_id)}</strong><br>${esc(crop.source)}<br>${esc(crop.class_name)} · ${Number(crop.confidence).toFixed(2)}<br><small>(${crop.crop_bbox.join(', ')})</small></figcaption>`;
    const label = document.createElement('label'); label.textContent = '标记 ';
    const input = document.createElement('input'); input.type = 'text'; input.maxLength = 80; input.value = crop.mark || '';
    input.placeholder = '可填写备注'; input.setAttribute('aria-label', `${crop.crop_id} 标记`);
    input.addEventListener('change', async () => {
      input.disabled = true;
      try {
        await api(`/api/head-cuts/${record.run_id}/marks`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ marks: { [crop.crop_id]: input.value } }) });
        crop.mark = input.value.trim();
      } catch (error) { showError(el('cutProgress'), error, '保存标记失败'); }
      finally { input.disabled = false; }
    });
    label.appendChild(input); figure.appendChild(label); grid.appendChild(figure);
  }
  el('cutPageInfo').textContent = `${page+1} / ${pages}`;
  el('cutPreviousPage').disabled = page === 0;
  el('cutNextPage').disabled = page+1 >= pages;
  if (record.errors.length) {
    const errors = document.createElement('p'); errors.className = 'status error';
    errors.textContent = record.errors.map(e => `${e.source}：${e.message}`).join('；'); grid.appendChild(errors);
  }
}

async function loadHeadCuts() {
  try {
    const images = await api('/api/images'); renderImages(images.images);
    await loadBatches();
  } catch (error) { showError(el('cutProgress'), error, '加载头切割失败'); }
}

el('cutRefresh').addEventListener('click', loadHeadCuts);
el('cutSelectAll').addEventListener('change', event => { selected.clear(); if (event.target.checked) filenames.forEach(n => selected.add(n)); syncSelection(); });
el('cutClearSelection').addEventListener('click', () => { selected.clear(); syncSelection(); });
el('cutBatchSelect').addEventListener('change', () => { if (el('cutBatchSelect').value) openBatch(el('cutBatchSelect').value); });
el('cutSourceFilter').addEventListener('change', () => { page = 0; renderHeads(); });
el('cutSearch').addEventListener('input', () => { page = 0; renderHeads(); });
el('cutPreviousPage').addEventListener('click', () => { page--; renderHeads(); });
el('cutNextPage').addEventListener('click', () => { page++; renderHeads(); });
el('cutCancel').addEventListener('click', async () => {
  try { if (activeJob) await api(`/api/jobs/${activeJob}/cancel`, { method: 'POST' }); }
  catch (error) { showError(el('cutProgress'), error, '取消任务失败'); }
});
el('cutInfer').addEventListener('click', async () => {
  if (activeJob || !selected.size) return;
  el('cutInfer').disabled = true;
  try {
    const mergeIou = Number(el('cutMergeIou').value || '.1');
    const mergeConfidence = Number(el('cutMergeConfidence').value || '.55');
    if ([mergeIou, mergeConfidence].some(v => !Number.isFinite(v) || v < 0 || v > 1)) throw new Error('IoU 和置信度门槛必须在0到1之间');
    const result = await api('/api/head-cuts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filenames: [...selected], rtdetr_base: !!el('cutRtdetrBase').checked, merge_iou: mergeIou, merge_confidence: mergeConfidence }) });
    activeJob = result.run_id; syncSelection(); await loadBatches(result.run_id);
  } catch (error) { showError(el('cutProgress'), error, '头切割失败'); syncSelection(); }
});

export { loadHeadCuts };
