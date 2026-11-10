import { el, esc, api } from './core.js?v=95';
import { loadImageGallery, gallerySelection } from './images.js?v=127';

let busy = false, activeJob = null, stopping = false;
let comparison = null;
const terminal = status => ['completed', 'cancelled', 'failed'].includes(status);

function pane(name) {
  document.querySelectorAll('#tab-mirror [data-mirror]').forEach(button => button.classList.toggle('active', button.dataset.mirror === name));
  for (const key of ['images', 'history', 'detail']) el(`mirror-pane-${key}`).classList.toggle('active', key === name);
  el('mirrorImageFolderMove').hidden = name !== 'images';
  el('mirrorRefresh').hidden = name === 'detail';
  if (name === 'history') loadHistory();
}

function sync() {
  const selection = gallerySelection('mirror');
  const locked = busy || selection.busy;
  el('mirrorRefresh').disabled = busy;
  el('mirrorStart').disabled = busy || selection.busy || !selection.selected.length;
  el('mirrorStop').hidden = !busy;
  el('mirrorStop').disabled = !activeJob || stopping;
  el('mirrorStop').textContent = stopping ? '正在停止…' : '停止推理';
  el('mirrorImageGrid').inert = busy;
  el('mirrorImageSelectAll').disabled = locked || !selection.count;
  el('mirrorImageClearSelection').disabled = locked || !selection.selected.length;
  el('mirrorImageDeleteSelected').disabled = locked || !selection.selected.length;
  el('mirrorImageFolderMove').disabled = locked;
}

function compare(row) {
  comparison = row;
  el('mirrorDisplayControls').hidden = false;
  el('mirrorResultName').textContent = row.relative_path || row.source;
  el('mirrorComparison').innerHTML = [
    ['原图', row.result], ['水平镜像', row.mirror_result],
  ].map(([label, result]) => `<figure>
    <figcaption><b>${label}</b><span>${result.count} 个目标 · ${esc(result.elapsed_seconds)}s</span></figcaption>
    <div class="viewer mirror-viewer"><img class="base" src="${esc(result.source_url)}" alt="${label}原始图片"><div class="boxes" data-mirror-boxes></div></div>
    </figure>`).join('');
  paintComparison();
}

function paintComparison() {
  if (!comparison) return;
  const visible = el('mirrorShowMarkers').checked;
  const points = el('mirrorShowPoints').checked;
  const labels = el('mirrorShowLabels').checked;
  el('mirrorShowPoints').disabled = !visible;
  el('mirrorShowLabels').disabled = !visible;
  const results = [comparison.result, comparison.mirror_result];
  el('mirrorComparison').querySelectorAll('[data-mirror-boxes]').forEach((layer, index) => {
    layer.replaceChildren();
    if (!visible) return;
    const result = results[index];
    for (const target of result.targets || []) {
      const [x1, y1, x2, y2] = target.bbox;
      const box = document.createElement('div');
      box.className = points ? 'bbox aggregate-point' : 'bbox';
      box.style.setProperty('--c', '#1a73e8');
      box.style.left = `${(points ? (x1 + x2) / 2 : x1) / result.image_width * 100}%`;
      box.style.top = `${(points ? (y1 + y2) / 2 : y1) / result.image_height * 100}%`;
      box.style.width = points ? '9px' : `${(x2 - x1) / result.image_width * 100}%`;
      box.style.height = points ? '9px' : `${(y2 - y1) / result.image_height * 100}%`;
      if (labels) {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = `${target.class_name || 'head'} ${target.confidence ?? ''}`;
        box.appendChild(tag);
      }
      layer.appendChild(box);
    }
  });
}

function progress(job) {
  activeJob = job.job_id;
  stopping = !!job.cancel_requested;
  sync();
  el('mirrorStatus').textContent = `${job.message} · 成功 ${job.completed_images} 组 · 失败 ${job.failed_images} 组`;
  el('mirrorProgress').innerHTML = `<div class="aggregate-progress" role="status">
    <b>${esc(job.source)} · ${job.image_index}/${job.image_total}</b>
    <label>${esc(job.stages.find(stage => stage.key === job.stage)?.label || '准备')} · ${job.percent || 0}%
      <progress max="100" value="${job.percent || 0}" aria-label="镜像推理进度"></progress></label>
    <div class="chips">${job.results.map((row,index) => `<button class="chip" data-mirror-row="${index}" ${row.status === 'completed' ? '' : 'disabled'}>${esc(row.source)} · ${esc(({completed:'查看对比',running:'处理中',waiting:'待处理',cancelled:'已停止',failed:'失败'})[row.status])}</button>`).join('')}</div>
    ${job.results.filter(row => row.status === 'failed').map(row => `<p class="action-reason">${esc(row.source)}：${esc(row.error)}</p>`).join('')}
    </div>`;
  el('mirrorProgress').querySelectorAll('[data-mirror-row]').forEach(button => button.addEventListener('click', () => compare(job.results[Number(button.dataset.mirrorRow)])));
}

async function follow(initial) {
  let job = initial, shown = false;
  try {
    for (;;) {
      progress(job);
      const first = job.results.find(row => row.status === 'completed');
      if (first && !shown) { compare(first); shown = true; }
      if (terminal(job.status)) break;
      await new Promise(resolve => setTimeout(resolve, 800));
      job = await api(`/api/aggregate/jobs/${job.job_id}`, {}, 15000);
    }
    if (!shown) {
      el('mirrorDisplayControls').hidden = true;
      el('mirrorComparison').textContent = job.status === 'cancelled' ? '推理已停止，本批次尚无完成的对比结果。' : '本批次没有成功的对比结果。';
    }
  } catch (error) { el('mirrorStatus').textContent = `查询镜像任务失败：${error.message}`; }
  finally { busy = false; activeJob = null; stopping = false; sync(); }
}

async function loadHistory() {
  try {
    const data = await api('/api/mirror/history');
    const relation = Number(el('mirrorCountFilter').value);
    const records = data.records.filter(row => Math.sign(row.count - row.mirror_count) === relation);
    el('mirrorHistoryStatus').textContent = `显示 ${records.length} 组 / 共 ${data.records.length} 组镜像结果`;
    el('mirrorHistoryGrid').innerHTML = records.map(row => `<figure data-mirror-record="${esc(row.id)}" title="查看原图和镜像对比">
      <img src="${esc(row.thumbnail)}" alt="${esc(row.source)}"><figcaption>${esc(row.relative_path || row.source)}<br>原图 ${row.count} 个 / 镜像 ${row.mirror_count} 个</figcaption></figure>`).join('') || '<p class="empty">暂无符合条件的镜像结果</p>';
    el('mirrorHistoryGrid').querySelectorAll('[data-mirror-record]').forEach(figure => figure.addEventListener('click', async () => {
      try {
        compare(await api(`/api/mirror/history/${figure.dataset.mirrorRecord}`));
        if (!busy) { el('mirrorProgress').textContent = ''; el('mirrorStatus').textContent = ''; }
        pane('detail');
      } catch (error) { el('mirrorHistoryStatus').textContent = error.message; }
    }));
  } catch (error) { el('mirrorHistoryStatus').textContent = error.message; }
}

async function loadMirror() {
  if (busy) return;
  await loadImageGallery('mirror');
  try {
    const { job } = await api('/api/mirror/active');
    if (job) {
      busy = true; pane('detail');
      el('mirrorComparison').textContent = '';
      el('mirrorDisplayControls').hidden = true;
      follow(job);
    }
  } catch (error) { el('mirrorStatus').textContent = error.message; }
  sync();
}

el('mirrorStart').addEventListener('click', async () => {
  const filenames = gallerySelection('mirror').selected;
  if (busy || !filenames.length) return;
  busy = true; stopping = false; sync(); pane('detail');
  el('mirrorComparison').textContent = '';
  el('mirrorDisplayControls').hidden = true;
  el('mirrorProgress').textContent = '';
  el('mirrorResultName').textContent = '';
  el('mirrorStatus').textContent = '正在准备镜像推理…';
  try {
    const job = await api('/api/mirror/batch', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filenames}) });
    await follow(job);
  } catch (error) { busy = false; sync(); el('mirrorStatus').textContent = `启动失败：${error.message}`; }
});
el('mirrorStop').addEventListener('click', async () => {
  if (!activeJob || stopping) return;
  stopping = true; sync();
  try { await api(`/api/aggregate/jobs/${activeJob}/cancel`, {method:'POST'}); }
  catch (error) { stopping = false; sync(); el('mirrorStatus').textContent = `停止失败：${error.message}`; }
});
el('mirrorRefresh').addEventListener('click', () => el('mirror-pane-history').classList.contains('active') ? loadHistory() : loadMirror());
el('mirrorCountFilter').addEventListener('change', loadHistory);
for (const id of ['mirrorShowMarkers', 'mirrorShowPoints', 'mirrorShowLabels']) {
  el(id).addEventListener('change', paintComparison);
}
document.querySelectorAll('#tab-mirror [data-mirror]').forEach(button => button.addEventListener('click', () => pane(button.dataset.mirror)));
document.addEventListener('gallery-selection-change', event => { if (event.detail.kind === 'mirror') sync(); });

export { loadMirror };
