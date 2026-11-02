import { el, esc } from './core.js?v=95';

let latest = null;
const numeric = values => values.filter(value => typeof value === 'number' && Number.isFinite(value));
const format = (value, suffix = '') => Number.isFinite(value) ? `${value.toFixed(2)}${suffix}` : '—';
const percentile = (values, fraction) => {
  const sorted = numeric(values).sort((a, b) => a - b);
  return sorted.length ? sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)] : NaN;
};

function renderExperiments(record) {
  latest = record;
  const sessions = record?.analysis_sessions || [];
  el('actExportExperiments').disabled = !sessions.length;
  el('actExperimentHistory').innerHTML = sessions.length ? [...sessions].reverse().map(session => {
    const results = session.results || [], measured = results.filter(item => !item.cached);
    const metrics = measured.map(item => item.action?.metrics || item.metrics || {});
    const times = metrics.map(item => item.generation_seconds);
    const peaks = numeric(metrics.map(item => item.peak_allocated_mib));
    const failed = results.filter(item => item.status === 'failed').length;
    const cached = results.filter(item => item.cached).length;
    const statuses = { completed: '已完成', cancelled: '已取消', failed: '失败', running: '运行中' };
    return `<details class="action-experiment-entry"><summary>${esc(session.decoding.label)} · ${esc(new Date(session.started_at).toLocaleString())} · ${esc(statuses[session.status] || session.status)} · 已处理 ${results.length}/${session.total} · 失败 ${failed} · 缓存 ${cached}</summary>
      <p class="hint">${esc(session.model)} · ${esc(session.quantization)} · 提示词 ${esc(session.prompt_config_id.slice(0, 8))} · ${session.decoding.num_beams} 束 · 上限 ${session.decoding.max_new_tokens} · 关闭采样</p>
      <p class="hint">总耗时 ${format(session.elapsed_seconds, 's')} · 生成中位数 ${format(percentile(times, .5), 's')} · P95 ${format(percentile(times, .95), 's')} · 分配显存峰值 ${format(peaks.length ? Math.max(...peaks) : NaN, ' MiB')} · 响应校验失败 ${results.filter(item => item.raw_response != null).length}</p>
      <table><thead><tr><th>目标 / 输入摘要</th><th>结果与可见证据</th><th>生成耗时 / 输出 Token</th></tr></thead><tbody>${results.map(item => {
        const metric = item.action?.metrics || item.metrics || {};
        return `<tr><td>${esc(item.crop_id)}<br>${esc(item.source)}<br><small>${esc(item.input_sha256?.slice(0, 8) || '—')} · ${esc(item.region_bbox.join(', '))}</small></td>
          <td>${esc(item.action?.state || '分析失败')}${item.cached ? '（缓存）' : ''}<br>${esc(item.action?.evidence || item.error || '')}${item.raw_response != null ? `<details><summary>模型原始响应</summary><pre>${esc(item.raw_response)}</pre></details>` : ''}</td>
          <td>${item.cached ? '缓存不计时' : `${format(metric.generation_seconds, 's')} / ${metric.generated_tokens ?? '—'}`}<br>${metric.cold_load ? `含冷加载 ${format(metric.model_load_seconds, 's')}` : ''}</td></tr>`;
      }).join('')}</tbody></table></details>`;
  }).join('') : '<p class="hint">分析后显示实验记录；旧批次的既有结果没有解码实验记录。</p>';
}

el('actExportExperiments').addEventListener('click', () => {
  if (!latest?.analysis_sessions?.length) return;
  const value = { run_id: latest.run_id, prompt_profiles: latest.prompt_profiles, analysis_sessions: latest.analysis_sessions };
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = `action-experiments-${latest.run_id}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

export { renderExperiments };
