import { el, esc } from './core.js?v=95';

let active = null;
let busy = false;
const dialog = el('aggFolderDialog');
// One picker is shared by pending images and saved results.
document.body.appendChild(dialog);

function render() {
  const choices = active.choices();
  let target = el('aggFolderTarget').value;
  if (!choices.some(folder => folder.id === target)) target = choices[0]?.id || '';
  el('aggFolderTarget').value = target;
  el('aggFolderChoices').innerHTML = choices.map(folder => `<label class="agg-folder-choice">
    <input type="radio" name="aggFolderChoice" value="${esc(folder.id)}" ${folder.id === target ? 'checked' : ''}>
    <span>${esc(folder.name)}</span><small>${folder.count} 个文件</small></label>`).join('');
  el('aggFolderChoices').querySelectorAll('input').forEach(input => input.addEventListener('change', () => {
    el('aggFolderTarget').value = input.value;
  }));
  sync();
}

function sync() {
  for (const id of ['aggFolderCreate', 'aggFolderName', 'aggFolderCancel']) el(id).disabled = busy;
  el('aggFolderConfirm').disabled = busy || !active?.canMove;
  el('aggFolderChoices').querySelectorAll('input').forEach(input => { input.disabled = busy; });
}

function openFolderDialog(options) {
  if (busy || dialog.open) return;
  active = options;
  el('aggFolderDialogStatus').textContent = options.description;
  el('aggFolderName').value = '';
  el('aggFolderTarget').value = '';
  render();
  dialog.showModal();
}

el('aggFolderCancel').addEventListener('click', () => dialog.close());
dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
el('aggFolderCreate').addEventListener('click', async () => {
  if (busy) return;
  busy = true; sync();
  try {
    const folder = await active.create(el('aggFolderName').value);
    el('aggFolderName').value = '';
    el('aggFolderTarget').value = folder.id;
    render();
    el('aggFolderDialogStatus').textContent = `已创建文件夹“${folder.name}”`;
  } catch (error) { el('aggFolderDialogStatus').textContent = error.message; }
  finally { busy = false; sync(); }
});
el('aggFolderConfirm').addEventListener('click', async () => {
  if (busy || !active?.canMove) return;
  busy = true; sync();
  try {
    await active.move(el('aggFolderTarget').value);
    dialog.close();
  } catch (error) { el('aggFolderDialogStatus').textContent = error.message; }
  finally { busy = false; sync(); }
});

export { openFolderDialog };
