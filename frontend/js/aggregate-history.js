import { openFolderDialog } from './folder-dialog.js?v=124';
import { el, esc, api } from './core.js?v=95';

const selected = new Set();
const selectedFolders = new Set();
let records = [], folders = [], busy = false, openRecord, currentFolder = null;
const post = data => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

function branchIds(root) {
  const found = new Set([root]);
  const pending = [root];
  while (pending.length) {
    const parent = pending.pop();
    for (const folder of folders) {
      if (folder.parent_id === parent && !found.has(folder.id)) { found.add(folder.id); pending.push(folder.id); }
    }
  }
  return found;
}

function folderPath(id) {
  const names = [], visited = new Set();
  while (id && !visited.has(id)) {
    visited.add(id);
    const folder = folders.find(item => item.id === id);
    if (!folder) break;
    names.unshift(folder.name); id = folder.parent_id;
  }
  return names.join(' / ');
}

function childFolders() {
  return folders.filter(folder => (folder.parent_id || null) === currentFolder);
}

function folderCount(id) {
  const branch = branchIds(id);
  const members = new Set(folders.filter(folder => branch.has(folder.id)).flatMap(folder => folder.record_ids));
  return records.filter(row => members.has(row.id)).length;
}

function visibleRecords() {
  const assigned = new Set(folders.flatMap(folder => folder.record_ids));
  const folder = folders.find(item => item.id === currentFolder);
  return records.filter(row => folder ? folder.record_ids.includes(row.id) : !assigned.has(row.id));
}

function sync() {
  const visible = visibleRecords();
  const total = visible.length + childFolders().length;
  const count = selected.size + selectedFolders.size;
  el('aggHistoryCount').textContent = `已选 ${selected.size} 条记录、${selectedFolders.size} 个文件夹`;
  el('aggHistoryAll').checked = total > 0 && count === total;
  el('aggHistoryAll').indeterminate = count > 0 && count < total;
  el('aggHistoryAll').disabled = busy || !total;
  el('aggHistoryClear').disabled = busy || !count;
  el('aggHistoryDelete').disabled = busy || !count;
  el('aggHistoryExport').disabled = busy || !count;
  el('aggExportCompression').disabled = busy;
  el('aggFolderMove').disabled = busy;
  el('aggFolderBack').disabled = busy;
  el('aggHistoryGrid').inert = busy;
  document.dispatchEvent(new CustomEvent('aggregate-history-selection-change'));
  el('aggHistoryGrid').querySelectorAll('figure').forEach(figure => {
    const checked = figure.dataset.folder ? selectedFolders.has(figure.dataset.folder) : selected.has(figure.dataset.record);
    figure.classList.toggle('is-selected', checked);
    figure.querySelector('input').checked = checked;
  });
}

function folderChoices() {
  const assigned = new Set(folders.flatMap(folder => folder.record_ids));
  const excluded = new Set([...selectedFolders].flatMap(id => [...branchIds(id)]));
  const choices = [...folders.filter(folder => !excluded.has(folder.id)).map(folder => ({ ...folder, name: folderPath(folder.id), count: folderCount(folder.id) })),
    { id: '', name: '根目录 / 未分类', count: records.filter(row => !assigned.has(row.id)).length }];
  return choices;
}

async function loadAggregateHistory() {
  try {
    const [history, categories] = await Promise.all([api('/api/aggregate/history'), api('/api/aggregate/folders')]);
    records = history.records; folders = categories.folders;
    if (currentFolder && !folders.some(folder => folder.id === currentFolder)) currentFolder = null;
    const available = new Set(visibleRecords().map(row => row.id));
    for (const id of selected) if (!available.has(id)) selected.delete(id);
    for (const id of selectedFolders) if (!folders.some(folder => folder.id === id)) selectedFolders.delete(id);
    const current = folders.find(folder => folder.id === currentFolder);
    el('aggFolderLocation').textContent = current ? folderPath(current.id) : '全部文件夹 / 未分类结果';
    el('aggFolderBack').hidden = !current;
    el('aggFolderBreadcrumb').hidden = !current;
    const folderCards = childFolders().map(folder => `<figure data-folder="${folder.id}" title="单击选择，双击打开文件夹 ${esc(folder.name)}">
      <div class="agg-folder-preview" aria-hidden="true">📁</div>
      <figcaption>${esc(folder.name)}<br>${folderCount(folder.id)} 条结果</figcaption>
      <label class="selection-check"><input type="checkbox" aria-label="选择文件夹 ${esc(folder.name)}"></label>
      </figure>`).join('');
    el('aggHistoryGrid').innerHTML = folderCards + visibleRecords().map(row => `<figure data-record="${esc(row.id)}" title="单击选择，双击查看已保存结果">
      <img src="${esc(row.thumbnail)}" alt="${esc(row.source)}">
      <figcaption>${esc(row.source)}<br>${row.count} 个目标<br>${esc(row.created_at ? new Date(row.created_at).toLocaleString() : '')}</figcaption>
      <label class="selection-check"><input type="checkbox" aria-label="选择记录 ${esc(row.source)}"></label>
      </figure>`).join('') || '<div class="empty">暂无已保存的聚合结果</div>';
    el('aggHistoryGrid').querySelectorAll('figure').forEach(figure => {
      const folderId = figure.dataset.folder;
      const id = folderId || figure.dataset.record;
      const selection = folderId ? selectedFolders : selected;
      figure.querySelector('label').addEventListener('click', event => event.stopPropagation());
      figure.querySelector('label').addEventListener('dblclick', event => event.stopPropagation());
      figure.querySelector('input').addEventListener('change', event => {
        if (event.target.checked) selection.add(id); else selection.delete(id);
        sync();
      });
      figure.addEventListener('click', event => {
        if (busy || event.detail > 1) return;
        if (selection.has(id)) selection.delete(id); else selection.add(id);
        sync();
      });
      figure.addEventListener('dblclick', async () => {
        if (busy) return;
        if (folderId) {
          currentFolder = folderId; selected.clear(); selectedFolders.clear();
          await loadAggregateHistory();
          return;
        }
        try { await openHistoryRecord(id, visibleRecords().map(row => row.id)); }
        catch (error) { el('aggHistoryStatus').textContent = error.message; }
      });
    });
    sync();
  } catch (error) { el('aggHistoryStatus').textContent = error.message; }
}

async function openHistoryRecord(id, ids) {
  const index = ids.indexOf(id);
  const row = await api(`/api/aggregate/history/${encodeURIComponent(id)}`);
  openRecord(row, {
    canPrevious: () => index > 0,
    canNext: () => index < ids.length - 1,
    previous: () => openHistoryRecord(ids[index - 1], ids),
    next: () => openHistoryRecord(ids[index + 1], ids),
  });
}

function setupAggregateHistory(onOpen) {
  openRecord = onOpen;
  el('aggFolderBack').addEventListener('click', () => {
    currentFolder = folders.find(folder => folder.id === currentFolder)?.parent_id || null; selected.clear(); selectedFolders.clear(); loadAggregateHistory();
  });
  el('aggFolderMove').addEventListener('click', () => {
    if (busy) return;
    openFolderDialog({
      description: `已选 ${selected.size} 条结果、${selectedFolders.size} 个文件夹`,
      canMove: !!(selected.size || selectedFolders.size),
      choices: folderChoices,
      create: async name => {
        const folder = await api('/api/aggregate/folders', post({ name, parent_id: currentFolder }));
        await loadAggregateHistory();
        return folder;
      },
      move: async target => {
        busy = true; sync();
        try {
          await api('/api/aggregate/folders/move', post({ record_ids: [...selected], folder_ids: [...selectedFolders], folder_id: target || null }));
          selected.clear(); selectedFolders.clear(); await loadAggregateHistory();
          el('aggHistoryStatus').textContent = '所选记录已移动';
        } finally { busy = false; sync(); }
      },
    });
  });
  el('aggHistoryAll').addEventListener('change', event => {
    selected.clear(); selectedFolders.clear();
    if (event.target.checked) {
      visibleRecords().forEach(row => selected.add(row.id));
      childFolders().forEach(folder => selectedFolders.add(folder.id));
    }
    sync();
  });
  el('aggHistoryClear').addEventListener('click', () => { selected.clear(); selectedFolders.clear(); sync(); });
  el('aggHistoryDelete').addEventListener('click', async () => {
    if (busy || (!selected.size && !selectedFolders.size)) return;
    const removingFolders = new Set([...selectedFolders].flatMap(id => [...branchIds(id)]));
    const removingRecords = new Set(selected);
    folders.filter(folder => removingFolders.has(folder.id)).forEach(folder => {
      folder.record_ids.forEach(id => removingRecords.add(id));
    });
    const recordCount = records.filter(row => removingRecords.has(row.id)).length;
    if (!confirm(`删除 ${removingFolders.size} 个文件夹（含子文件夹）及 ${recordCount} 条结果记录？文件夹内的结果也会从列表删除，原批次快照仍保留。`)) return;
    busy = true; sync();
    try {
      const result = await api('/api/aggregate/history/delete', post({ record_ids: [...selected], folder_ids: [...selectedFolders] }));
      selected.clear(); selectedFolders.clear();
      await loadAggregateHistory();
      el('aggHistoryStatus').textContent = `已删除 ${result.deleted_folders.length} 个文件夹、${result.deleted.length} 条结果记录`;
    } catch (error) { el('aggHistoryStatus').textContent = error.message; }
    finally { busy = false; sync(); }
  });
  el('aggHistoryExport').addEventListener('click', async () => {
    if (busy || (!selected.size && !selectedFolders.size)) return;
    const folder = selectedFolders.size === 1 && !selected.size ? folders.find(item => selectedFolders.has(item.id)) : null;
    busy = true; sync();
    try {
      const response = await fetch('/api/aggregate/history/export', post({ record_ids: [...selected], folder_ids: [...selectedFolders], compression: el('aggExportCompression').value }));
      if (!response.ok) throw new Error((await response.json()).detail || '导出失败');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url; link.download = folder ? `${folder.name}.zip` : 'aggregate-results.zip';
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      el('aggHistoryStatus').textContent = `已导出 ZIP（${(blob.size / 1024 / 1024).toFixed(2)} MB），内含 JSON 数据和 HTML 查看器。解压后打开 HTML 即可查看，也可手动导入 JSON。`;
    } catch (error) { el('aggHistoryStatus').textContent = error.message; }
    finally { busy = false; sync(); }
  });
}

function aggregateHistorySelection() {
  return { record_ids: [...selected], folder_ids: [...selectedFolders], busy };
}

export { setupAggregateHistory, loadAggregateHistory, aggregateHistorySelection };
