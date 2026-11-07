import { el, esc, api } from './core.js?v=95';

let saved = null, defaults = null, jsonDirty = false, onSaved = () => {};
let mode = 'actions';
const savedProfiles = {}, drafts = {};
const endpoint = (suffix = '') => `/api/actions/prompt${suffix}?mode=${mode}`;
const payload = config => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) });
const message = (text, failed = false) => { el('actPromptStatus').textContent = text; el('actPromptStatus').style.color = failed ? '#b3261e' : '#527082'; };

function getPromptStates() {
  return [...new Set([...(savedProfiles.actions?.actions || []).filter(item => item.enabled).map(item => item.name), '抬头', '低头', '疑似抬头', '疑似低头', '无法判断'])];
}

function readForm() {
  return { version: 1, instructions: el('actPromptInstructions').value,
    actions: [...el('actPromptRules').querySelectorAll('tr')].map(row => ({
      name: row.querySelector('[data-cfg="name"]').value.trim(),
      enabled: row.querySelector('[data-cfg="enabled"]').checked,
      priority: Number(row.querySelector('[data-cfg="priority"]').value),
      rule: row.querySelector('[data-cfg="rule"]').value.trim(),
    })) };
}

function dirty() {
  jsonDirty = false; el('actPromptJson').value = JSON.stringify(readForm(), null, 2);
  message('未保存；保存后用于下一次分析，运行中的任务继续使用启动时的配置。');
}

function addRow(item) {
  const row = document.createElement('tr');
  row.innerHTML = `<td><input data-cfg="enabled" type="checkbox" aria-label="启用动作 ${esc(item.name)}"></td>
    <td><input data-cfg="name" maxlength="30" aria-label="动作名称" placeholder="例如：打哈欠"></td>
    <td><input data-cfg="priority" type="number" min="0" max="100" step="1" aria-label="动作优先级"></td>
    <td><textarea data-cfg="rule" rows="2" maxlength="1200" aria-label="动作判定规则" placeholder="描述需要看见的姿态或物品关系"></textarea></td>
    <td><button class="btn small ghost" type="button" aria-label="删除动作 ${esc(item.name)}">删除</button></td>`;
  row.querySelector('[data-cfg="enabled"]').checked = item.enabled;
  row.querySelector('[data-cfg="name"]').value = item.name;
  row.querySelector('[data-cfg="priority"]').value = item.priority;
  row.querySelector('[data-cfg="rule"]').value = item.rule;
  if (mode === 'head') {
    row.querySelector('[data-cfg="name"]').readOnly = true;
    row.querySelector('[data-cfg="enabled"]').disabled = true;
    row.querySelector('button').disabled = true;
  }
  row.querySelectorAll('input, textarea').forEach(input => input.addEventListener('input', dirty));
  row.querySelector('button').addEventListener('click', () => { row.remove(); dirty(); });
  el('actPromptRules').appendChild(row);
}

function renderForm(config) {
  if (!config || !Array.isArray(config.actions)) throw new Error('JSON 中需要 actions 动作数组');
  el('actPromptInstructions').value = config.instructions || '';
  el('actPromptRules').innerHTML = '';
  config.actions.forEach(item => addRow(item));
  el('actPromptJson').value = JSON.stringify(config, null, 2); jsonDirty = false;
}

function draft() {
  const config = jsonDirty ? JSON.parse(el('actPromptJson').value) : readForm();
  if (!config || config.version !== 1 || !Array.isArray(config.actions) || !config.actions.length || config.actions.length > 30) {
    throw new Error('配置需要 version: 1 和 1～30 个动作');
  }
  const names = new Set();
  for (const item of config.actions) {
    if (typeof item.name !== 'string' || !item.name.trim() || item.name.length > 30 || item.name.trim() === '无法判断') throw new Error('动作名称不能为空、不能超过 30 字或使用保留状态“无法判断”');
    const name = item.name.trim().toLowerCase();
    if (names.has(name)) throw new Error(`动作名称重复：${item.name}`);
    names.add(name);
    if (typeof item.rule !== 'string' || !item.rule.trim() || item.rule.length > 1200) throw new Error(`请填写 ${item.name} 的判定规则（最多 1200 字）`);
    if (!Number.isInteger(item.priority) || item.priority < 0 || item.priority > 100) throw new Error('优先级必须是 0～100 的整数');
    if (typeof item.enabled !== 'boolean') throw new Error('enabled 必须为 true 或 false');
  }
  if (!config.actions.some(item => item.enabled)) throw new Error('至少启用一个动作');
  if (mode === 'head' && (config.actions.length !== 2 || !['抬头', '低头'].every(name => config.actions.some(a => a.name === name && a.enabled)))) throw new Error('Head 模式固定启用抬头、低头，请编辑说明、规则和优先级');
  return config;
}

async function loadActionPrompts(force = false) {
  if (saved && !force) return saved;
  const data = await api(endpoint());
  saved = data.config; defaults = data.defaults;
  savedProfiles[mode] = saved;
  el('actPromptAdd').disabled = mode === 'head';
  renderForm(saved); el('actPromptPreview').value = data.compiled_prompt;
  message(`已加载配置 ${data.config_id.slice(0, 8)} · ${data.states.length-1} 个启用动作 · 无法判断始终保留`);
  onSaved(); return saved;
}

async function operation(work) {
  el('actPromptMode').disabled = true;
  try { await work(); } catch (err) { message(err.message || String(err), true); }
  finally { el('actPromptMode').disabled = false; }
}

function initPromptEditor(callback) {
  onSaved = callback;
  el('actPromptMode').addEventListener('change', () => operation(async () => {
    drafts[mode] = { config: readForm(), json: el('actPromptJson').value, jsonDirty };
    const previous = mode;
    mode = el('actPromptMode').value;
    try { await loadActionPrompts(true); }
    catch (error) { mode = previous; el('actPromptMode').value = previous; throw error; }
    if (drafts[mode]) {
      renderForm(drafts[mode].config);
      el('actPromptJson').value = drafts[mode].json;
      jsonDirty = drafts[mode].jsonDirty;
      message('已切换配置，保留该模式的编辑内容；点击保存后用于下一次分析。');
    }
  }));
  el('actPromptInstructions').addEventListener('input', dirty);
  el('actPromptJson').addEventListener('input', () => { jsonDirty = true; message('JSON 已修改，保存配置或应用到表单后生效。'); });
  el('actPromptApplyJson').addEventListener('click', () => operation(async () => { const config = draft(); renderForm(config); dirty(); }));
  el('actPromptReload').addEventListener('click', () => operation(() => loadActionPrompts(true)));
  el('actPromptDefaults').addEventListener('click', () => {
    if (defaults) { renderForm(defaults); dirty(); }
  });
  el('actPromptAdd').addEventListener('click', () => operation(async () => {
    if (jsonDirty) renderForm(draft());
    addRow({ name: '', enabled: true, priority: 50, rule: '' }); dirty();
  }));
  el('actPromptSave').addEventListener('click', () => operation(async () => {
    const config = draft();
    const data = await api(endpoint(), { ...payload(config), method: 'PUT' });
    saved = data.config; defaults = data.defaults; renderForm(saved);
    savedProfiles[mode] = saved;
    delete drafts[mode];
    el('actPromptPreview').value = data.compiled_prompt;
    message(`已保存配置 ${data.config_id.slice(0, 8)} · ${data.states.length-1} 个启用动作；下一次分析使用新规则。`);
    onSaved();
  }));
  el('actPromptBuild').addEventListener('click', () => operation(async () => {
    const data = await api(endpoint('/preview'), payload(draft()));
    el('actPromptPreview').value = data.compiled_prompt;
    message('已根据当前编辑内容生成预览；尚未保存的配置不会用于推理。');
  }));
}

export { getPromptStates, initPromptEditor, loadActionPrompts };
