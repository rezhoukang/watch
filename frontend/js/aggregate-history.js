import { el, esc, api } from './core.js?v=95';

const selected = new Set();
let records = [], busy = false, openRecord;
const request = ids => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ record_ids: ids }) });

function sync() {
  el('aggHistoryCount').textContent = `已选 ${selected.size} 条`;
  el('aggHistoryAll').checked = records.length > 0 && selected.size === records.length;
  el('aggHistoryAll').indeterminate = selected.size > 0 && selected.size < records.length;
  el('aggHistoryAll').disabled = busy || !records.length;
  for (const id of ['aggHistoryClear', 'aggHistoryDelete', 'aggHistoryExport']) el(id).disabled = busy || !selected.size;
  el('aggHistoryGrid').inert = busy;
  el('aggHistoryGrid').querySelectorAll('figure').forEach(figure => {
    const checked = selected.has(figure.dataset.record);
    figure.classList.toggle('is-selected', checked);
    figure.querySelector('input').checked = checked;
  });
}

async function loadAggregateHistory() {
  try {
    records = (await api('/api/aggregate/history')).records;
    const available = new Set(records.map(row => row.id));
    for (const id of selected) if (!available.has(id)) selected.delete(id);
    el('aggHistoryGrid').innerHTML = records.map(row => `<figure data-record="${esc(row.id)}" title="查看已保存结果">
      <img src="${esc(row.thumbnail)}" alt="${esc(row.source)}">
      <figcaption>${esc(row.source)}<br>${row.count} 个目标<br>${esc(row.created_at ? new Date(row.created_at).toLocaleString() : '')}</figcaption>
      <label class="selection-check"><input type="checkbox" aria-label="选择记录 ${esc(row.source)}"></label>
      </figure>`).join('') || '<div class="empty">暂无已保存的聚合结果</div>';
    el('aggHistoryGrid').querySelectorAll('figure').forEach(figure => {
      const id = figure.dataset.record;
      figure.querySelector('label').addEventListener('click', event => event.stopPropagation());
      figure.querySelector('input').addEventListener('change', event => {
        if (event.target.checked) selected.add(id); else selected.delete(id);
        sync();
      });
      figure.addEventListener('click', async () => {
        if (busy) return;
        try { openRecord(await api(`/api/aggregate/history/${id}`)); }
        catch (error) { el('aggHistoryStatus').textContent = error.message; }
      });
    });
    sync();
  } catch (error) { el('aggHistoryStatus').textContent = error.message; }
}

function setupAggregateHistory(onOpen) {
  openRecord = onOpen;
  el('aggHistoryAll').addEventListener('change', event => {
    selected.clear();
    if (event.target.checked) records.forEach(row => selected.add(row.id));
    sync();
  });
  el('aggHistoryClear').addEventListener('click', () => { selected.clear(); sync(); });
  el('aggHistoryDelete').addEventListener('click', async () => {
    if (busy || !selected.size || !confirm(`从列表删除选中的 ${selected.size} 条聚合结果记录？原批次快照仍保留。`)) return;
    busy = true; sync();
    try {
      await api('/api/aggregate/history/delete', request([...selected]));
      selected.clear();
      await loadAggregateHistory();
      el('aggHistoryStatus').textContent = '所选记录已从结果列表删除';
    } catch (error) { el('aggHistoryStatus').textContent = error.message; }
    finally { busy = false; sync(); }
  });
  el('aggHistoryExport').addEventListener('click', async () => {
    if (busy || !selected.size) return;
    busy = true; sync();
    try {
      const response = await fetch('/api/aggregate/history/export', request([...selected]));
      if (!response.ok) throw new Error((await response.json()).detail || '导出失败');
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a');
      link.href = url; link.download = 'aggregate-results.zip';
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      el('aggHistoryStatus').textContent = '已导出 ZIP，内含 JSON 数据和 HTML 查看器。解压后打开 HTML 即可查看，也可手动导入 JSON。';
    } catch (error) { el('aggHistoryStatus').textContent = error.message; }
    finally { busy = false; sync(); }
  });
}

export { setupAggregateHistory, loadAggregateHistory };
