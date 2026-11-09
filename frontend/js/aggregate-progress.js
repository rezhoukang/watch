import { el, esc, api } from './core.js?v=95';

function renderAggregateProgress(job, filename) {
  const counts = {};
  for (const target of job.preview_targets || []) {
    const state = target.action?.state;
    if (state) counts[state] = (counts[state] || 0) + 1;
  }
  const preview = Object.entries(counts).map(([state, count]) => `${state} ${count}`).join(' · ');
  const stages = job.stages || [], stage = stages.find(item => item.key === job.stage);
  const percent = Math.max(0, Math.min(100, Number(job.percent) || 0));
  const known = typeof job.total === 'number';
  const seconds = Math.max(0, ['completed', 'failed', 'cancelled'].includes(job.status) ? job.elapsed_seconds || 0
    : Date.now() / 1000 - (job.created_at || Date.now() / 1000)).toFixed(1);
  const stageStatus = { completed: '完成', running: '进行中', waiting: '待处理', failed: '失败' };
  el('aggStatus').textContent = `${stage?.label || '准备'} · 已用时 ${seconds}s`;
  el('aggResult').innerHTML = `<div class="aggregate-progress" role="status" aria-live="polite">
    <b>${job.status === 'failed' ? '聚合失败 · ' : ''}${esc(stage?.label || '准备聚合任务')} · ${esc(job.message || '正在创建任务')}</b>
    <p class="hint">图片：${esc(filename)} · 已用时 ${seconds}s</p>
    ${preview ? `<p class="hint">当前已出结果（补充识别完成前为初步结果）：${esc(preview)}</p>` : ''}
    <label>阶段进度 ${percent}%<progress aria-label="聚合阶段进度" max="100" value="${percent}"></progress></label>
    <p class="hint">按已完成阶段与当前阶段进度计算，不代表耗时比例。</p>
    <div class="agg-steps">${stages.map(item => `<span class="agg-step ${item.status === 'running' ? 'is-running' : item.status === 'failed' ? 'is-failed' : item.status === 'waiting' ? 'is-off' : ''}">${esc(item.label)} · ${esc(stageStatus[item.status] || item.status)}</span>`).join('')}</div>
    <label>当前阶段：${job.stage === 'multimodal' && known ? `已处理 ${job.current}/${job.total} 个目标${job.failed ? ` · 失败 ${job.failed}` : ''}` : job.status === 'failed' ? '已停止' : known ? '已完成' : '处理中'}
      <progress aria-label="当前阶段进度" ${known ? `max="${Math.max(1, job.total)}" value="${job.total === 0 ? 1 : job.current}"` : 'max="1"'}></progress>
    </label></div>`;
}

async function followAggregateJob(initial, filename) {
  let job = initial;
  const ticker = setInterval(() => renderAggregateProgress(job, filename), 500);
  try {
    renderAggregateProgress(job, filename);
    while (!['completed', 'failed', 'cancelled'].includes(job.status)) {
      await new Promise(resolve => setTimeout(resolve, 800));
      job = await api(`/api/aggregate/jobs/${job.job_id}`, {}, 15000);
      renderAggregateProgress(job, filename);
    }
    if (job.status === 'cancelled') return null;
    if (job.status === 'failed') {
      const error = new Error(job.error || job.message || '聚合失败');
      error.job = job;
      throw error;
    }
    if (!job.result || !Array.isArray(job.result.targets)) throw new Error('聚合结果不完整，请重新打开任务');
    return job.result;
  } finally {
    clearInterval(ticker);
  }
}

export { renderAggregateProgress, followAggregateJob };
