import { el, esc, api } from './core.js?v=95';

function renderBatch(job, onSelect) {
  const panel = el('aggBatchPanel');
  panel.hidden = false;
  panel.innerHTML = `<div class="aggregate-progress" role="status">
    <b>批量聚合 ${job.image_index}/${job.image_total} · ${esc(job.message)}</b>
    <p class="hint">当前图片 ${esc(job.source)} · 成功 ${job.completed_images} 张 · 失败 ${job.failed_images} 张</p>
    <progress max="100" value="${job.percent || 0}" aria-label="批量推理进度"></progress>
    <div class="chips">${job.results.map((row, index) => `<button class="chip" data-batch-index="${index}" ${row.status === 'completed' ? '' : 'disabled'}>
      ${esc(row.source)} · ${esc(({waiting:'待处理',running:'处理中',completed:'查看结果',failed:'失败',cancelled:'已停止'})[row.status])}</button>`).join('')}</div>
    ${job.results.filter(row => row.status === 'failed').map(row => `<p class="action-reason">${esc(row.source)}：${esc(row.error)}</p>`).join('')}
    </div>`;
  panel.querySelectorAll('[data-batch-index]').forEach(button => button.addEventListener('click', () => {
    onSelect(Number(button.dataset.batchIndex));
  }));
}

async function followBatch(initial, onResult, onJob) {
  let job = initial, currentIndex = -1;
  function adjacent(direction) {
    const completed = job.results.flatMap((row, index) => row.status === 'completed' ? [index] : []);
    return completed[completed.indexOf(currentIndex) + direction];
  }
  function showResult(index) {
    const row = job.results[index];
    if (row?.status !== 'completed') return;
    currentIndex = index;
    onResult(row.result, row.source, {
      canPrevious: () => adjacent(-1) !== undefined,
      canNext: () => adjacent(1) !== undefined,
      previous: () => showResult(adjacent(-1)),
      next: () => showResult(adjacent(1)),
    });
  }
  const update = () => {
    onJob?.(job);
    renderBatch(job, showResult);
    if (currentIndex < 0) showResult(job.results.findIndex(row => row.status === 'completed'));
  };
  update();
  while (!['completed', 'failed', 'cancelled'].includes(job.status)) {
    await new Promise(resolve => setTimeout(resolve, 800));
    job = await api(`/api/aggregate/jobs/${job.job_id}`, {}, 15000);
    update();
  }
  if (currentIndex < 0) el('aggResult').textContent = job.status === 'cancelled' ? '推理已停止，本批次尚无已完成结果。' : '本批次没有成功的结果，请查看上方失败原因。';
  return job;
}

export { followBatch };
