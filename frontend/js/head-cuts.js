import { el, esc, api, showError } from './core.js?v=76';

const selected = new Set();
let filenames = [], current = null, activeJob = null, page = 0, loadVersion = 0;
const PAGE_SIZE = 60;
const fileUrl = (run, file) => `/api/head-cuts/${run}/files/${file.split('/').map(encodeURIComponent).join('/')}`;
const running = status => status === 'running' || status === 'queued';
const batchBusy = record => running(record.status) || running(record.face_detection?.status) || running(record.recognition?.status);

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
  if (current?.run_id !== record.run_id) { el('cutFaceFilter').value = ''; el('cutRecFilter').value = ''; }
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
  } else if (activeJob === runId) {
    activeJob = null; syncSelection();
  }
  if (batchBusy(record)) setTimeout(() => pollBatch(runId, version), 1200);
}

async function pollBatch(runId, version) {
  try {
    const record = await api(`/api/head-cuts/${runId}`);
    if (version !== loadVersion && (current?.run_id === runId || activeJob !== runId)) return;
    if (version === loadVersion) { current = record; renderHeads(); }
    if (batchBusy(record)) {
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
  el('cutDownload').hidden = batchBusy(record);
  el('cutDownload').href = `/api/head-cuts/${record.run_id}/download`;
  const faceJob = record.face_detection;
  el('cutDetectFaces').disabled = running(record.status) || running(faceJob?.status) || !record.crop_count;
  el('cutFaceConfidence').disabled = running(faceJob?.status);
  el('cutFaceCancel').hidden = !running(faceJob?.status);
  el('cutFaceProgress').textContent = faceJob
    ? `${faceJob.progress}/${faceJob.total} · ${faceJob.message} · 阈值 ${faceJob.config.confidence}` : '';
  const recJob = record.recognition;
  const anyDetected = record.crops.some(c => c.face_detection?.status === 'detected');
  el('cutRecognizeFaces').disabled = running(record.status) || running(faceJob?.status) || running(recJob?.status) || !anyDetected;
  for (const id of ['cutRecThreshold', 'cutRecConfidence', 'cutRecSize']) el(id).disabled = running(recJob?.status);
  el('cutRecCancel').hidden = !running(recJob?.status);
  el('cutFaceProgress').textContent += recJob
    ? `｜识别 ${recJob.progress}/${recJob.total} · ${recJob.message} · 阈值 ${recJob.config.threshold}` : '';
  const source = el('cutSourceFilter').value;
  const search = el('cutSearch').value.trim().toLowerCase();
  const faceFilter = el('cutFaceFilter').value;
  const recFilter = el('cutRecFilter').value;
  const recStatusOf = c => {
    if (!c.recognition) return 'pending';
    if (c.recognition.status === 'done') return c.recognition.name ? 'matched' : 'unknown';
    return c.recognition.status;
  };
  const filtered = record.crops.filter(c => (!source || c.source === source) &&
    (!faceFilter || (c.face_detection?.status || 'pending') === faceFilter) &&
    (!recFilter || recStatusOf(c) === recFilter) &&
    `${c.crop_id} ${c.class_name} ${c.recognition?.name || ''}`.toLowerCase().includes(search));
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
  grid.innerHTML = '';
  for (const crop of filtered.slice(page*PAGE_SIZE, (page+1)*PAGE_SIZE)) {
    const figure = document.createElement('figure');
    figure.innerHTML = `<a href="${esc(fileUrl(record.run_id, crop.raw_file))}" target="_blank"><img src="${esc(fileUrl(record.run_id, crop.marked_file))}" alt="${esc(crop.crop_id)}" loading="lazy"></a><figcaption><strong>${esc(crop.crop_id)}</strong><br>${esc(crop.source)}<br>${esc(crop.class_name)} · ${Number(crop.confidence).toFixed(2)}<br><small>(${crop.crop_bbox.join(', ')})</small></figcaption>`;
    const detection = crop.face_detection;
    const faceInfo = document.createElement('div'); faceInfo.className = 'head-face-info';
    const labels = { detected: '检出人脸', no_face: '未检出人脸', error: '检测失败', pending: '未检测' };
    faceInfo.textContent = detection ? `${labels[detection.status] || detection.status}${detection.face_count ? ` ${detection.face_count} 张` : ''}${detection.message ? `：${detection.message}` : ''}` : '未检测人脸';
    figure.appendChild(faceInfo);
    if (detection?.faces?.length) {
      const faces = document.createElement('div'); faces.className = 'head-face-strip';
      const recognition = crop.recognition;
      const recByFace = {};
      for (const item of (recognition?.faces || [])) recByFace[item.face_id] = item;
      const sizeNote = rec => (rec?.face_pixels ? `｜人脸 ${rec.face_pixels}px` : '');
      // 一对一消解：被别的脸抢走的候选，在提示里说明是谁抢的
      const conflictNote = rec => {
        const skipped = rec?.conflict?.skipped || [];
        if (!skipped.length) return '';
        return '｜被抢占的候选：' + skipped.map(s => `${s.name}(${s.similarity}) 给了 ${s.taken_by}`).join('；');
      };
      faces.innerHTML = detection.faces.map((face, index) => {
        const rec = recByFace[face.face_id];
        let badge = '';
        if (rec?.status === 'matched') {
          const rank = rec.match?.rank || 1;
          const rankTag = rank > 1 ? ` <i class="rec-rank">备选#${rank}</i>` : '';
          const rankNote = rank > 1 ? `｜第 1 候选被抢占，顺延到第 ${rank} 候选` : '';
          badge = `<em class="rec-badge rec-hit" title="相似度 ${rec.match?.similarity}${rankNote}${conflictNote(rec)}${sizeNote(rec)}">${esc(rec.match?.name || '')} ${rec.match?.similarity}${rankTag}</em>`;
        } else if (rec?.status === 'rejected') {
          badge = `<em class="rec-badge rec-skip" title="${esc((rec.reasons || []).join('；'))}${sizeNote(rec)}">质量不足</em>`;
        } else if (rec?.status === 'unknown') {
          const why = rec.match?.reason === 'taken'
            ? `最高候选 ${rec.match?.name} 已被得分更高的另一张脸占用，且没有可用备选`
            : '最高候选相似度不足';
          badge = `<em class="rec-badge" title="${esc(why + conflictNote(rec))}｜最高 ${rec.match?.similarity}${sizeNote(rec)}">未识别</em>`;
        } else if (rec?.status === 'error') {
          badge = `<em class="rec-badge rec-skip" title="${esc(rec.message || '')}">识别失败</em>`;
        }
        const query = `<a class="rec-query" href="${esc(fileUrl(record.run_id, face.file))}" target="_blank"><small title="${esc(face.face_id)}">人脸 ${index+1} · ${Number(face.confidence).toFixed(2)}</small>${badge}</a>`;
        // 命中时把人像库里的对应登记照放在右侧，便于人工核对是否同一个人
        const lib = (rec?.status === 'matched' && rec.match?.photo_url)
          ? `<span class="rec-pair-arrow" aria-hidden="true">→</span><a class="rec-lib" href="${esc(rec.match.photo_url)}" target="_blank" title="人像库登记照：${esc(rec.match.name || '')}（编号 ${esc(rec.match.entry_id || '')}）"><img src="${esc(rec.match.photo_url)}" alt="${esc(rec.match.name || '')} 登记照" loading="lazy"><em>库内登记照</em></a>`
          : '';
        return `<div class="rec-item">${query}${lib}</div>`;
      }).join('');
      figure.appendChild(faces);
      if (detection.annotated_file) {
        const link = document.createElement('a'); link.className = 'head-face-link'; link.textContent = '查看人脸框';
        link.href = fileUrl(record.run_id, detection.annotated_file); link.target = '_blank'; figure.appendChild(link);
      }
    }
    grid.appendChild(figure);
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
el('cutFaceFilter').addEventListener('change', () => { page = 0; renderHeads(); });
el('cutRecFilter').addEventListener('change', () => { page = 0; renderHeads(); });
el('cutDetectFaces').addEventListener('click', async () => {
  if (!current) return;
  const runId = current.run_id;
  el('cutDetectFaces').disabled = true;
  try {
    const confidence = Number(el('cutFaceConfidence').value || '.5');
    if (!Number.isFinite(confidence) || confidence < .01 || confidence > .99) throw new Error('人脸置信度必须在0.01到0.99之间');
    await api(`/api/head-cuts/${runId}/faces`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confidence }) });
    if (current?.run_id === runId) await openBatch(runId);
  } catch (error) { showError(el('cutFaceProgress'), error, '人脸检测失败'); el('cutDetectFaces').disabled = false; }
});
el('cutRecognizeFaces').addEventListener('click', async () => {
  if (!current) return;
  const runId = current.run_id;
  el('cutRecognizeFaces').disabled = true;
  try {
    const threshold = Number(el('cutRecThreshold').value || '.4');
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('识别阈值必须在0到1之间');
    const minConfidence = Number(el('cutRecConfidence').value || '.5');
    if (!Number.isFinite(minConfidence) || minConfidence < 0 || minConfidence > 1) throw new Error('质量置信度门槛必须在0到1之间');
    const minSize = Number(el('cutRecSize').value || '24');
    if (!Number.isInteger(minSize) || minSize < 1 || minSize > 1000) throw new Error('最小人脸像素必须是1到1000之间的整数');
    await api(`/api/head-cuts/${runId}/recognize`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold, min_confidence: minConfidence, min_size: minSize }) });
    if (current?.run_id === runId) await openBatch(runId);
  } catch (error) { showError(el('cutFaceProgress'), error, '1:N 识别失败'); el('cutRecognizeFaces').disabled = false; }
});
el('cutRecCancel').addEventListener('click', async () => {
  try { if (current?.recognition?.job_id) await api(`/api/jobs/${current.recognition.job_id}/cancel`, { method: 'POST' }); }
  catch (error) { showError(el('cutFaceProgress'), error, '取消识别失败'); }
});
el('cutFaceCancel').addEventListener('click', async () => {
  try { if (current?.face_detection?.job_id) await api(`/api/jobs/${current.face_detection.job_id}/cancel`, { method: 'POST' }); }
  catch (error) { showError(el('cutFaceProgress'), error, '取消人脸检测失败'); }
});
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
