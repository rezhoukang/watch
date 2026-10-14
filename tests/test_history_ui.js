// Exercise actual history/merge controller code with a small DOM fixture.
// This checks selection and requests; it does not replace browser visual QA.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.dataset = {}; this.events = {};
    this.value = ''; this.disabled = false; this.classes = new Set();
    this.classList = { toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name) };
  }
  replaceChildren() { this.children = []; this.value = ''; }
  appendChild(child) {
    this.children.push(child);
    if (this.tag === 'select' && this.children.length === 1) this.value = child.value;
  }
  addEventListener(name, handler) { this.events[name] = handler; }
  setAttribute() {}
  removeAttribute(name) { delete this[name]; }
  fire(name) { return this.events[name]({ target: this, stopPropagation() {} }); }
}

const nodes = new Map();
const el = (id) => {
  if (!nodes.has(id)) nodes.set(id, new Element(['mergeBase', 'mergeIncoming'].includes(id) ? 'select' : 'div'));
  return nodes.get(id);
};
const record = (id, hash) => ({ record_id: id, input_sha256: hash, source: `${hash}.png`,
  created_at: '2026-10-14T00:00:00+08:00', weight: 'test.pt', mode: 'yolo', kind: 'inference',
  count: 1, parameters: { confidence: 0.38 }, imgsz: 640, result_url: `/result/${id}` });
let records = [record('a', 'image1'), record('b', 'image1'), record('c', 'image2')];
const requests = [];
let rendered = null;
const context = vm.createContext({
  el, console, Date, Set, encodeURIComponent, esc: (x) => String(x),
  document: { createElement: (tag) => new Element(tag),
    querySelectorAll: () => el('inferenceHistoryGrid').children },
  switchSubPane: (pane) => assert.equal(pane, 'result'),
  renderDetectResult: (...args) => { rendered = args; },
  showError: (_, error) => { throw error; },
  api: async (url, options = {}) => {
    requests.push({ url, options });
    if (url === '/api/inferences/merge') {
      const merged = { ...record('merged', 'image1'), kind: 'merge', mode: 'merged' };
      records = [merged, ...records]; return merged;
    }
    if (url === '/api/inferences') return { records };
    return records.find((r) => url.endsWith('/' + r.record_id));
  },
});
const source = fs.readFileSync(path.join(__dirname, '../frontend/app.js'), 'utf8');
const start = source.indexOf('let INFERENCE_RECORDS = [];');
const end = source.indexOf('async function loadVideos()', start);
vm.runInContext(source.slice(start, end), context);

(async () => {
  await vm.runInContext('loadInferenceHistory()', context);
  assert.equal(el('inferenceHistoryGrid').children.length, 3);
  assert.equal(el('mergeBase').value, 'a');
  assert.equal(el('mergeIncoming').value, 'b');
  assert.equal(el('mergeIncoming').children.length, 1);
  assert.equal(el('mergeInferenceBtn').disabled, false);
  el('swapMerge').fire('click');
  assert.equal(el('mergeBase').value, 'b');
  assert.equal(el('mergeIncoming').value, 'a');
  el('mergeBase').value = 'c'; el('mergeBase').fire('change');
  assert.equal(el('mergeIncoming').value, '');
  assert.equal(el('mergeInferenceBtn').disabled, true);
  assert.equal(el('mergeIncomingPreview').hidden, true);
  el('mergeBase').value = 'b'; el('mergeBase').fire('change');
  await el('mergeInferenceBtn').fire('click');
  const merge = requests.find((request) => request.url === '/api/inferences/merge');
  assert.deepEqual(JSON.parse(merge.options.body), { base_id: 'b', incoming_id: 'a' });
  assert.equal(rendered[0].record_id, 'merged');
  assert.equal(rendered[2], 0.01); // Merge view displays all saved frames initially.
  assert.equal(el('inferenceHistoryGrid').children.length, 4);
  assert.equal(requests.some((request) => request.url.startsWith('/api/detect/')), false);
  console.log('History UI controller checks passed: compatibility, swap direction, merge, saved-record reopening.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
