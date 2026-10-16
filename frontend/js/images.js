import { el, esc, api, banner, showError } from './core.js?v=89';
import { readDetectOptions } from './models.js?v=89';
import { bindImageHover, hideHoverPreview } from './preview.js?v=89';
import { switchSubPane } from './navigation.js?v=89';
import { renderDetectResult } from './viewer.js?v=89';
import { importBehaviorRecords } from './annotator.js?v=89';

/* ==================== 图片推理 ==================== */

const GALLERY_STATE = {
  images: { grid: 'imageGrid', prefix: 'image', selected: new Set(), items: [], busy: false },
  aggregate: { grid: 'aggImageGrid', prefix: 'aggImage', selected: new Set(), items: [], busy: false },
  history: { grid: 'inferenceHistoryGrid', prefix: 'history', selected: new Set(), items: [], busy: false },
};
let ACTIVE_INFERENCE_ID = null;

function syncGallerySelection(kind) {
  const state = GALLERY_STATE[kind];
  const busy = state.busy || (kind === 'history' && MERGE_BUSY);
  const all = el(`${state.prefix}SelectAll`);
  all.checked = state.items.length > 0 && state.selected.size === state.items.length;
  all.indeterminate = state.selected.size > 0 && state.selected.size < state.items.length;
  all.disabled = busy || !state.items.length;
  el(`${state.prefix}DeleteSelected`).disabled = busy || !state.selected.size;
  el(`${state.prefix}ClearSelection`).disabled = busy || !state.selected.size;
  if (kind === 'history') el('historyBehaviorImport').disabled = busy || !state.selected.size;
  el(`${state.prefix}SelectionCount`).textContent = state.busy ? '正在删除…' : `已选 ${state.selected.size} 张`;
  document.querySelectorAll(`#${state.grid} figure`).forEach((figure) => {
    const selected = state.selected.has(figure.dataset.item);
    figure.classList.toggle('is-selected', selected);
    const check = figure.querySelector('[data-gallery-select]');
    if (check) { check.checked = selected; check.disabled = busy; }
    const del = figure.querySelector('.del-btn');
    if (del) del.disabled = busy;
  });
}

function setGalleryItems(kind, items) {
  const state = GALLERY_STATE[kind];
  state.items = items;
  state.selected = new Set([...state.selected].filter((item) => items.includes(item)));
}

function addGalleryControls(figure, kind, item, name) {
  figure.dataset.item = item;
  const label = document.createElement('label'); label.className = 'selection-check';
  const check = document.createElement('input'); check.type = 'checkbox'; check.dataset.gallerySelect = '';
  check.setAttribute('aria-label', `选择${name}`);
  label.addEventListener('click', (ev) => ev.stopPropagation());
  check.addEventListener('change', () => {
    const selected = GALLERY_STATE[kind].selected;
    if (check.checked) selected.add(item); else selected.delete(item);
    syncGallerySelection(kind);
  });
  label.appendChild(check); figure.appendChild(label);
  const del = document.createElement('button');
  del.type = 'button'; del.className = 'del-btn'; del.textContent = '×'; del.title = `删除${name}`;
  del.setAttribute('aria-label', del.title);
  del.addEventListener('click', (ev) => { ev.stopPropagation(); return deleteGalleryItems(kind, [item]); });
  figure.appendChild(del);
}

async function deleteGalleryItems(kind, items) {
  const state = GALLERY_STATE[kind];
  if (state.busy || (kind === 'history' && MERGE_BUSY) || !items.length) return;
  hideHoverPreview();
  const pending = kind !== 'history';
  const message = pending
    ? `删除选中的 ${items.length} 张待推理图片？已保存的历史推理记录仍保留。`
    : `删除选中的 ${items.length} 条推理记录及其保存的图片？已另存的合并结果仍保留。`;
  if (!confirm(message)) return;
  state.busy = true; syncGallerySelection(kind);
  if (kind === 'history') syncMergeChoices();
  try {
    const response = await api(`/api/${pending ? 'images' : 'inferences'}/delete`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pending ? { filenames: items } : { record_ids: items }),
    });
    const removed = new Set([...response.deleted, ...response.missing]);
    removed.forEach((item) => state.selected.delete(item));
    if (kind === 'history' && removed.has(ACTIVE_INFERENCE_ID)) {
      ACTIVE_INFERENCE_ID = null; el('detResultName').textContent = '';
      el('detectResult').textContent = '此推理记录已删除，请从图库选择其他图片或记录。';
    }
    await (pending ? state.reload() : loadInferenceHistory());
    if (response.failed.length) banner(`${response.failed.length} 项删除失败：${response.failed.map((f) => f.detail).join('；')}`);
  } catch (error) { banner('删除失败：' + error.message); }
  finally {
    state.busy = false; syncGallerySelection(kind);
    if (kind === 'history') syncMergeChoices();
  }
}

async function loadImages() {
  return loadImageGallery('images', detectImage);
}

async function loadImageGallery(kind, onImageClick) {
  const state = GALLERY_STATE[kind];
  state.reload = () => loadImageGallery(kind, onImageClick);
  const box = el(state.grid);
  try {
    const data = await api('/api/images');
    hideHoverPreview();
    setGalleryItems(kind, data.images.map((img) => img.filename));
    if (!data.images.length) {
      box.innerHTML = '<div class="empty">暂无图片，请先上传</div>';
      syncGallerySelection(kind);
      return;
    }
    box.innerHTML = '';
    data.images.forEach((img) => {
      const fig = document.createElement('figure');
      fig.title = (kind === 'aggregate' ? '点击聚合：' : '点击推理：') + img.filename;
      fig.innerHTML = `<img src="${esc(img.url)}" alt=""><figcaption>${esc(img.filename)}</figcaption>`;

      addGalleryControls(fig, kind, img.filename, img.filename);
      bindImageHover(fig, () => ({ src: img.url, name: img.filename }));
      fig.addEventListener('click', () => {
        if (state.busy) return;
        hideHoverPreview();
        return onImageClick(img.filename);
      });
      box.appendChild(fig);
    });
    syncGallerySelection(kind);
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

async function detectImage(filename) {
  const box = el('detectResult');
  ACTIVE_INFERENCE_ID = null;
  let ticker = null;
  const t0 = Date.now();

  try {
    // 1) 先取参数（元素缺失会被 catch 住并显示出来）
    const headDetection = !!el('detHeadDetection').checked;
    const imgsz = headDetection ? '640' : el('detImgsz').value || '640';
    const { weights, conf, isWorld, classes, isRtdetr, isPose, rtdetrBase, headDownRThreshold, headUpRThreshold, poseRoi, roiConfidence, mergeIou = 0, mergeConfidence = 0 } = readDetectOptions('det');
    const modeText = headDetection ? `目标检测｜${rtdetrBase ? 'RT-DETRv4 0.25 基准 + medium 0.45 补齐' : 'medium 0.45 基准 + RT-DETRv4 0.25 补齐'}｜IoU ≤ ${mergeIou} 补齐` : isRtdetr ? 'RT-DETRv4-S C-RADIOv4｜人头计数' : isWorld
      ? `YOLO-World${classes ? `｜prompt「${classes}」` : '｜prompt=权重内置'}`
      : isPose ? 'YOLO Pose 人体骨架' : '普通 YOLO';

    // 切到「推理结果」子页面，并显示当前图片名
    switchSubPane('result');
    el('detResultName').textContent = filename;

    // 2) 实时计时，区分"在跑"和"卡死"
    const paint = () => {
      box.innerHTML =
        `<div>推理中… <b>${((Date.now() - t0) / 1000).toFixed(1)}s</b></div>` +
        `<div style="font-size:12px;color:#80868b;margin-top:6px">` +
        `图片：${esc(filename)}｜置信度 ${esc(conf)}｜imgsz ${esc(imgsz)}｜${esc(modeText)}<br>` +
        `（第一次推理要加载模型；World 切换类别还要编码文字；超过 300 秒自动报超时）</div>`;
    };
    paint();
    ticker = setInterval(paint, 200);

    // 3) 请求后端（图片推理允许长耗时）
    const url =
      `/api/detect/image/${encodeURIComponent(filename)}` +
      `?conf=${encodeURIComponent(conf)}&weights=${encodeURIComponent(weights)}` +
      `&imgsz=${encodeURIComponent(imgsz)}` +
      (isPose ? `&head_down_r_threshold=${encodeURIComponent(headDownRThreshold)}&head_up_r_threshold=${encodeURIComponent(headUpRThreshold)}&pose_roi=${poseRoi}&roi_confidence=${encodeURIComponent(roiConfidence)}` : '') +
      (headDetection ? `&head_detection=true&merge_iou=${encodeURIComponent(mergeIou)}&merge_confidence=${encodeURIComponent(mergeConfidence)}` : '') +
      (rtdetrBase ? '&rtdetr_base=true' : '') +
      (classes ? `&classes=${encodeURIComponent(classes)}` : '');
    const r = await api(url, { method: 'POST' }, 300000);

    // 4) 渲染结果（字段缺失时明确报错，而不是静默空白）
    if (!r || !Array.isArray(r.detections)) {
      throw new Error(`后端返回数据不完整：${JSON.stringify(r).slice(0, 200)}`);
    }
    if (headDetection && (r.mode !== 'head_fusion' || !r.fusion)) {
      throw new Error('目标检测未生效：服务返回了普通推理结果。请重启 start.bat 并按 Ctrl+F5 刷新，然后重新推理。');
    }
    if (rtdetrBase && !r.fusion?.rtdetr_base) throw new Error('RT-DETR 基准未生效，请重启服务后刷新页面。');
    if (headDetection && Number(r.fusion?.merge_iou ?? 0) !== mergeIou) throw new Error('合并 IoU 阈值未生效，请重启服务后刷新页面。');
    if (headDetection && Number(r.fusion?.merge_confidence ?? 0) !== mergeConfidence) throw new Error('合并置信度门槛未生效，请重启服务后刷新页面。');
    if (ticker !== null) { clearInterval(ticker); ticker = null; }
    renderImageResult(r, filename, headDetection ? 0.01 : conf, (Date.now() - t0) / 1000, imgsz);
    loadInferenceHistory();

    // 刷新图库缩略图（失败不影响已显示的结果）
    try {
      await loadImages();
    } catch (e) {
      console.warn('[loadImages] 刷新失败', e);
    }
  } catch (err) {
    showError(box, err, '推理失败');
  } finally {
    // 无论成功失败都停掉计时器，避免"一直推理中"
    if (ticker !== null) clearInterval(ticker);
  }
}

el('refreshImages').addEventListener('click', loadImages);
Object.keys(GALLERY_STATE).forEach((kind) => {
  const state = GALLERY_STATE[kind];
  el(`${state.prefix}SelectAll`).addEventListener('change', (ev) => {
    state.selected = ev.target.checked ? new Set(state.items) : new Set(); syncGallerySelection(kind);
  });
  el(`${state.prefix}ClearSelection`).addEventListener('click', () => { state.selected.clear(); syncGallerySelection(kind); });
  el(`${state.prefix}DeleteSelected`).addEventListener('click', () => deleteGalleryItems(kind, [...state.selected]));
});

/* ==================== 历史推理图库与定向合并 ==================== */

let INFERENCE_RECORDS = [];
let MERGE_BUSY = false;

function recordConfidence(record) {
  const values = new Set();
  const visit = (params) => {
    if (!params || typeof params !== 'object') return;
    if (typeof params.confidence === 'number' && Number.isFinite(params.confidence)) values.add(params.confidence);
    visit(params.base); visit(params.incoming);
  };
  visit(record.parameters);
  return values.size ? [...values].join(' / ') : '未记录';
}

function recordLabel(record, role = '') {
  const time = new Date(record.created_at).toLocaleString('zh-CN');
  const model = record.fusion ? `目标检测：${record.fusion.rtdetr_base ? 'RT-DETRv4 基准 + medium' : 'medium 基准 + RT-DETRv4'}` : record.kind === 'merge' ? '合并结果' : (record.weight || record.mode);
  const color = role === 'base' ? '绿色（基准）' : role === 'incoming' ? '橙色（待合并）'
    : record.kind === 'merge' ? '基准绿 / 新增橙' : '绿色';
  return `${record.source}｜${model}｜${record.count} 个｜置信度 ${recordConfidence(record)}｜${time} #${record.record_id.slice(0, 8)}｜颜色 ${color}`;
}

function syncMergeChoices(preferredIncoming = null) {
  const baseSel = el('mergeBase');
  const incomingSel = el('mergeIncoming');
  const oldIncoming = preferredIncoming || incomingSel.value;
  const base = INFERENCE_RECORDS.find((r) => r.record_id === baseSel.value);
  incomingSel.replaceChildren();
  const compatible = base ? INFERENCE_RECORDS.filter((r) => r.input_sha256 === base.input_sha256 && r.record_id !== base.record_id) : [];
  compatible.forEach((record) => {
    const option = document.createElement('option'); option.value = record.record_id; option.textContent = recordLabel(record, 'incoming');
    incomingSel.appendChild(option);
  });
  if (compatible.some((r) => r.record_id === oldIncoming)) incomingSel.value = oldIncoming;
  if (!compatible.length) {
    const option = document.createElement('option'); option.value = ''; option.textContent = '暂无同一原图的其他推理记录';
    incomingSel.appendChild(option);
  }
  const busy = MERGE_BUSY || GALLERY_STATE.history.busy;
  baseSel.disabled = busy || !INFERENCE_RECORDS.length;
  incomingSel.disabled = busy || !compatible.length;
  const incoming = compatible.find((r) => r.record_id === incomingSel.value);
  baseSel.title = base ? recordLabel(base, 'base') : '';
  incomingSel.title = incoming ? recordLabel(incoming, 'incoming') : '';
  [['mergeBasePreview', base], ['mergeIncomingPreview', incoming]].forEach(([id, record]) => {
    const image = el(id); image.hidden = !record;
    if (record) { image.src = record.result_url; image.title = recordLabel(record, id === 'mergeBasePreview' ? 'base' : 'incoming'); }
    else image.removeAttribute('src');
  });
  el('mergeInferenceBtn').disabled = busy || !base || !incoming;
  el('swapMerge').disabled = busy || !base || !incoming;
  document.querySelectorAll('#inferenceHistoryGrid figure').forEach((figure) => {
    figure.classList.toggle('is-base', figure.dataset.id === base?.record_id);
    figure.classList.toggle('is-incoming', figure.dataset.id === incoming?.record_id);
  });
}

async function openInferenceRecord(recordId) {
  hideHoverPreview(); ACTIVE_INFERENCE_ID = null;
  switchSubPane('result');
  const box = el('detectResult'); box.textContent = '正在加载历史推理记录…';
  try {
    const record = await api(`/api/inferences/${encodeURIComponent(recordId)}`);
    el('detResultName').textContent = `${record.source} · ${new Date(record.created_at).toLocaleString('zh-CN')} #${record.record_id.slice(0, 8)}`;
    renderImageResult(record, record.source, record.kind === 'merge' ? 0.01 : record.parameters.confidence,
      record.elapsed_seconds || 0, record.imgsz);
  } catch (error) { showError(box, error, '历史记录加载失败'); }
}

function filteredInferenceRecords() {
  const kind = el('historyKindFilter').value || 'all';
  const weight = el('historyWeightFilter').value || 'all';
  const name = el('historyNameSearch').value.trim().toLocaleLowerCase('zh-CN');
  return INFERENCE_RECORDS.filter((record) => {
    const recordKind = record.kind === 'merge' ? 'merge' : 'inference';
    if (kind !== 'all' && recordKind !== kind) return false;
    if (weight !== 'all' && (recordKind === 'merge' || String(record.weight || record.mode) !== weight)) return false;
    return !name || String(record.source || '').toLocaleLowerCase('zh-CN').includes(name);
  });
}

function renderInferenceHistoryGallery() {
  el('historyWeightFilter').disabled = el('historyKindFilter').value === 'merge';
  const records = filteredInferenceRecords();
  setGalleryItems('history', records.map((record) => record.record_id));
  el('inferenceHistoryStatus').textContent = INFERENCE_RECORDS.length
    ? `共 ${INFERENCE_RECORDS.length} 条记录，当前显示 ${records.length} 条。点击缩略图查看保存的结果与参数。绿色边框为基准，橙色边框为待合并。`
    : '暂无记录。每次图片推理完成后自动保存原图、带框图、参数和坐标。';
  const grid = el('inferenceHistoryGrid'); grid.replaceChildren();
  if (!records.length) {
    grid.innerHTML = `<div class="empty">${INFERENCE_RECORDS.length ? '没有符合筛选条件的记录' : '暂无推理结果图片'}</div>`;
  } else {
    records.forEach((record) => {
      const figure = document.createElement('figure'); figure.dataset.id = record.record_id;
      figure.tabIndex = 0; figure.setAttribute('role', 'button'); figure.title = recordLabel(record);
      figure.innerHTML = `<img src="${esc(record.result_url)}" alt="${esc(record.source)} 推理结果" loading="lazy">` +
        `<figcaption>${esc(record.source)}<br>${esc(record.fusion ? `目标检测：${record.fusion.rtdetr_base ? 'RT-DETRv4' : 'medium'} 基准` : record.kind === 'merge' ? '合并结果' : record.weight)} · ${esc(record.count)} 个<br>` +
        `置信度 ${esc(recordConfidence(record))} · imgsz ${esc(record.imgsz)}<br>` +
        `${esc(new Date(record.created_at).toLocaleString('zh-CN'))} #${esc(record.record_id.slice(0, 8))}<br>` +
        `${record.kind === 'merge' ? '颜色：基准绿 / 新增橙' : '颜色：绿色'}</figcaption>`;
      addGalleryControls(figure, 'history', record.record_id, record.source + ' #' + record.record_id.slice(0, 8));
      bindImageHover(figure, () => ({ src: record.result_url, name: recordLabel(record) }));
      figure.addEventListener('click', () => openInferenceRecord(record.record_id));
      figure.addEventListener('keydown', (e) => { if (e.target === figure && e.key === 'Enter') openInferenceRecord(record.record_id); });
      const actions = document.createElement('div'); actions.className = 'history-card-actions';
      [['基准', 'base'], ['待合并', 'incoming']].forEach(([label, role]) => {
        const button = document.createElement('button'); button.className = 'btn small ghost'; button.textContent = `设为${label}`;
        button.addEventListener('click', (e) => {
          e.stopPropagation();
          if (MERGE_BUSY || GALLERY_STATE.history.busy) return;
          if (role === 'base') { el('mergeBase').value = record.record_id; syncMergeChoices(); }
          else {
            const base = INFERENCE_RECORDS.find((r) => r.record_id === el('mergeBase').value);
            if (!base || base.record_id === record.record_id || base.input_sha256 !== record.input_sha256) {
              el('mergeStatus').textContent = '请先选择同一原图的另一条记录作为基准。'; return;
            }
            syncMergeChoices(record.record_id);
          }
          el('mergeStatus').textContent = '';
        });
        actions.appendChild(button);
      });
      figure.appendChild(actions); grid.appendChild(figure);
    });
  }
  syncGallerySelection('history');
  syncMergeChoices();
}

function refreshHistoryWeightFilter() {
  const select = el('historyWeightFilter');
  const previous = select.value || 'all';
  const weights = [...new Set(INFERENCE_RECORDS
    .filter((record) => record.kind !== 'merge')
    .map((record) => String(record.weight || record.mode || '未记录')))].sort((a, b) => a.localeCompare(b, 'zh-CN'));
  select.replaceChildren();
  const all = document.createElement('option'); all.value = 'all'; all.textContent = '全部权重'; select.appendChild(all);
  weights.forEach((weight) => {
    const option = document.createElement('option'); option.value = weight; option.textContent = weight; select.appendChild(option);
  });
  select.value = weights.includes(previous) ? previous : 'all';
}

async function loadInferenceHistory() {
  try {
    const data = await api('/api/inferences');
    INFERENCE_RECORDS = data.records || [];
    refreshHistoryWeightFilter();
    const select = el('mergeBase'); const previous = select.value; select.replaceChildren();
    INFERENCE_RECORDS.forEach((record) => {
      const option = document.createElement('option'); option.value = record.record_id; option.textContent = recordLabel(record, 'base');
      select.appendChild(option);
    });
    if (INFERENCE_RECORDS.some((r) => r.record_id === previous)) select.value = previous;
    select.disabled = !INFERENCE_RECORDS.length;
    renderInferenceHistoryGallery();
  } catch (error) { el('inferenceHistoryStatus').textContent = '记录加载失败：' + error.message; }
}

el('refreshInferenceHistory').addEventListener('click', loadInferenceHistory);
el('historyKindFilter').addEventListener('change', () => {
  if (el('historyKindFilter').value === 'merge') el('historyWeightFilter').value = 'all';
  renderInferenceHistoryGallery();
});
el('historyWeightFilter').addEventListener('change', renderInferenceHistoryGallery);
el('historyNameSearch').addEventListener('input', renderInferenceHistoryGallery);
el('historyClearFilters').addEventListener('click', () => {
  el('historyKindFilter').value = 'all';
  el('historyWeightFilter').value = 'all';
  el('historyNameSearch').value = '';
  renderInferenceHistoryGallery();
});
[['mergeBasePreview', 'mergeBase', 'base'], ['mergeIncomingPreview', 'mergeIncoming', 'incoming']].forEach(([imageId, selectId, role]) => {
  bindImageHover(el(imageId), () => {
    const record = INFERENCE_RECORDS.find((r) => r.record_id === el(selectId).value);
    return record ? { src: record.result_url, name: recordLabel(record, role) } : null;
  });
});
el('mergeBase').addEventListener('change', () => { el('mergeStatus').textContent = ''; syncMergeChoices(); });
el('mergeIncoming').addEventListener('change', () => syncMergeChoices());
el('swapMerge').addEventListener('click', () => {
  const oldBase = el('mergeBase').value; const oldIncoming = el('mergeIncoming').value;
  if (!oldBase || !oldIncoming) return;
  el('mergeBase').value = oldIncoming; syncMergeChoices(oldBase);
  el('mergeStatus').textContent = '已交换合并方向。';
});
el('mergeInferenceBtn').addEventListener('click', async () => {
  if (MERGE_BUSY || GALLERY_STATE.history.busy) return;
  MERGE_BUSY = true; syncMergeChoices();
  syncGallerySelection('history');
  const status = el('mergeStatus'); status.textContent = '正在合并…';
  try {
    const result = await api('/api/inferences/merge', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_id: el('mergeBase').value, incoming_id: el('mergeIncoming').value, merge_iou: Number(el('mergeIou').value || 0), merge_confidence: Number(el('mergeConfidence').value || 0) }) });
    await loadInferenceHistory();
    await openInferenceRecord(result.record_id);
    status.textContent = '合并完成，已另存为新记录。';
  } catch (error) { status.textContent = '合并失败：' + error.message; }
  finally { MERGE_BUSY = false; syncMergeChoices(); syncGallerySelection('history'); }
});


function setActiveInference(recordId) { ACTIVE_INFERENCE_ID = recordId; }

el('historyBehaviorImport').addEventListener('click', async () => {
  const button = el('historyBehaviorImport'); button.disabled = true;
  try { await importBehaviorRecords([...GALLERY_STATE.history.selected]); }
  catch (error) { banner(error instanceof Error ? error.message : String(error)); }
  finally { syncGallerySelection('history'); }
});

function renderImageResult(record, filename, confidence, elapsed, imgsz) {
  renderDetectResult(record, filename, confidence, elapsed, imgsz, {
    onRecordOpen: setActiveInference,
    onRecordDelete: (id) => deleteGalleryItems('history', [id]),
  });
}

export { loadImages, loadImageGallery, detectImage, loadInferenceHistory };
