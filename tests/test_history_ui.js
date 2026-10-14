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
  querySelector(selector) {
    const all = this.children.flatMap((child) => [child, ...child.descendants()]);
    return selector === '[data-gallery-select]' ? all.find((child) => Object.hasOwn(child.dataset, 'gallerySelect')) : null;
  }
  descendants() { return this.children.flatMap((child) => [child, ...child.descendants()]); }
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
let detectionResponse = null;
let hovered = null;
let allowConfirm = true;
let bannerText = '';
let images = [{ filename: 'one.png', url: '/one.png' }, { filename: 'two.png', url: '/two.png' }];
const context = vm.createContext({
  el, console, Date, Set, encodeURIComponent, esc: (x) => String(x),
  setInterval: () => 1, clearInterval() {},
  readDetectOptions: () => ({ weights: 'medium.pt', classes: '', isWorld: false, isRtdetr: false }),
  confirm: () => allowConfirm, hideHoverPreview: () => { hovered = null; },
  showHoverPreview: (src, name) => { hovered = { src, name }; }, placeHoverPreview() {}, hoverEv: null,
  banner: (message) => { bannerText = message; },
  document: { createElement: (tag) => new Element(tag),
    querySelectorAll: (selector) => selector.includes('#imageGrid') ? el('imageGrid').children : el('inferenceHistoryGrid').children },
  switchSubPane: (pane) => assert.equal(pane, 'result'),
  renderDetectResult: (...args) => { rendered = args; },
  showError: (_, error) => { throw error; },
  api: async (url, options = {}) => {
    requests.push({ url, options });
    if (url.startsWith('/api/detect/image/')) return detectionResponse;
    if (url === '/api/inferences/delete') {
      const items = JSON.parse(options.body).record_ids;
      records = records.filter((r) => !items.includes(r.record_id));
      return { deleted: items, missing: [], failed: [] };
    }
    if (url === '/api/images/delete') {
      const items = JSON.parse(options.body).filenames;
      images = images.filter((r) => !items.includes(r.filename));
      return { deleted: items, missing: [], failed: [] };
    }
    if (url === '/api/images') return { images };
    if (url === '/api/inferences/merge') {
      const merged = { ...record('merged', 'image1'), kind: 'merge', mode: 'merged' };
      records = [merged, ...records]; return merged;
    }
    if (url === '/api/inferences') return { records };
    return records.find((r) => url.endsWith('/' + r.record_id));
  },
});
const source = fs.readFileSync(path.join(__dirname, '../frontend/js/images.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
assert.equal(html.includes('id="deleteMergeBase"'), false);
assert.equal(html.includes('id="deleteMergeIncoming"'), false);
const previewSource = fs.readFileSync(path.join(__dirname, '../frontend/js/preview.js'), 'utf8');
const hoverStart = previewSource.indexOf('function bindImageHover(');
vm.runInContext(previewSource.slice(hoverStart, previewSource.indexOf('// 图片下载完', hoverStart)), context);
const start = source.indexOf('const GALLERY_STATE = {');
vm.runInContext(source.slice(start).replace(/^export .*$/gm, ''), context);

(async () => {
  await vm.runInContext('loadInferenceHistory()', context);
  assert.equal(el('inferenceHistoryGrid').children.length, 3);
  assert.equal(el('mergeBase').value, 'a');
  assert.equal(el('mergeIncoming').value, 'b');
  assert.equal(el('mergeIncoming').children.length, 1);
  assert.equal(el('mergeInferenceBtn').disabled, false);
  assert.equal(el('historyWeightFilter').children.length, 2);
  el('historyNameSearch').value = 'image2'; el('historyNameSearch').fire('input');
  assert.equal(el('inferenceHistoryGrid').children.length, 1);
  el('historySelectAll').checked = true; el('historySelectAll').fire('change');
  assert.equal(vm.runInContext('GALLERY_STATE.history.selected.size', context), 1);
  el('historyClearFilters').fire('click');
  assert.equal(el('inferenceHistoryGrid').children.length, 3);
  assert.equal(vm.runInContext('GALLERY_STATE.history.selected.size', context), 1);
  el('historyClearSelection').fire('click');
  const label = el('mergeBase').children[0].textContent;
  assert(label.indexOf('置信度 0.38') < label.indexOf('2026'));
  assert(label.indexOf('2026') < label.indexOf('#a'));
  assert(label.endsWith('颜色 绿色（基准）'));
  assert(el('mergeIncoming').children[0].textContent.endsWith('颜色 橙色（待合并）'));
  el('inferenceHistoryGrid').children[0].fire('mouseenter');
  assert.equal(hovered.src, '/result/a');
  el('mergeBasePreview').fire('mouseenter');
  assert.equal(hovered.src, '/result/a');
  assert.equal(vm.runInContext("recordConfidence({parameters:{base:{confidence:0.38},incoming:{base:{confidence:0.25},incoming:{confidence:0.38}}}})", context), '0.38 / 0.25');
  el('swapMerge').fire('click');
  assert.equal(el('mergeBase').value, 'b');
  assert.equal(el('mergeIncoming').value, 'a');
  el('mergeBase').value = 'c'; el('mergeBase').fire('change');
  assert.equal(el('mergeIncoming').value, '');
  assert.equal(el('mergeInferenceBtn').disabled, true);
  assert.equal(el('mergeIncomingPreview').hidden, true);
  el('mergeBase').value = 'b'; el('mergeBase').fire('change');
  el('mergeIou').value = '0.3';
  el('mergeConfidence').value = '0.5';
  await el('mergeInferenceBtn').fire('click');
  const merge = requests.find((request) => request.url === '/api/inferences/merge');
  assert.deepEqual(JSON.parse(merge.options.body), { base_id: 'b', incoming_id: 'a', merge_iou: 0.3, merge_confidence: 0.5 });
  assert.equal(rendered[0].record_id, 'merged');
  assert.equal(rendered[2], 0.01); // Merge view displays all saved frames initially.
  assert.equal(el('inferenceHistoryGrid').children.length, 4);
  el('historyKindFilter').value = 'merge'; el('historyKindFilter').fire('change');
  assert.equal(el('inferenceHistoryGrid').children.length, 1);
  el('historyKindFilter').value = 'inference'; el('historyKindFilter').fire('change');
  assert.equal(el('inferenceHistoryGrid').children.length, 3);
  el('historyWeightFilter').value = 'test.pt'; el('historyWeightFilter').fire('change');
  assert.equal(el('inferenceHistoryGrid').children.length, 3);
  el('historyClearFilters').fire('click');
  assert.equal(el('inferenceHistoryGrid').children.length, 4);
  assert.equal(requests.some((request) => request.url.startsWith('/api/detect/')), false);
  const firstCard = el('inferenceHistoryGrid').children[0];
  const check = firstCard.querySelector('[data-gallery-select]'); check.checked = true; check.fire('change');
  assert.equal(el('historySelectAll').indeterminate, true);
  assert.equal(vm.runInContext('GALLERY_STATE.history.selected.size', context), 1);
  allowConfirm = false;
  await el('historyDeleteSelected').fire('click');
  assert.equal(records.length, 4);
  allowConfirm = true;
  vm.runInContext("ACTIVE_INFERENCE_ID = 'merged'", context);
  await el('historyDeleteSelected').fire('click');
  assert.equal(records.length, 3);
  assert.equal(vm.runInContext('ACTIVE_INFERENCE_ID', context), null);
  assert.equal(vm.runInContext('GALLERY_STATE.history.selected.size', context), 0);
  el('historySelectAll').checked = true; el('historySelectAll').fire('change');
  assert.equal(vm.runInContext('GALLERY_STATE.history.selected.size', context), 3);
  await el('historyDeleteSelected').fire('click');
  assert.equal(records.length, 0);
  assert.equal(el('mergeBasePreview').hidden, true);
  assert.equal(el('mergeInferenceBtn').disabled, true);
  await vm.runInContext('loadImages()', context);
  el('imageGrid').children[0].fire('mouseenter'); assert.equal(hovered.src, '/one.png');
  el('imageSelectAll').checked = true; el('imageSelectAll').fire('change');
  assert.equal(vm.runInContext('GALLERY_STATE.images.selected.size', context), 2);
  el('imageClearSelection').fire('click');
  assert.equal(vm.runInContext('GALLERY_STATE.images.selected.size', context), 0);
  el('imageSelectAll').checked = true; el('imageSelectAll').fire('change');
  await el('imageDeleteSelected').fire('click');
  assert.equal(images.length, 0);
  assert.equal(el('imageSelectAll').disabled, true);
  assert.equal(bannerText, '');
  el('detHeadDetection').checked = true;
  detectionResponse = { mode: 'yolo', detections: [] };
  await assert.rejects(vm.runInContext("detectImage('test.png')", context), /head检测未生效/);
  detectionResponse = { mode: 'head_fusion', detections: [], fusion: { added_count: 0 } };
  await vm.runInContext("detectImage('test.png')", context);
  assert.equal(rendered[0].mode, 'head_fusion');
  assert.equal(rendered[2], 0.01);
  const detectionRequests = requests.filter((request) => request.url.startsWith('/api/detect/image/'));
  assert.equal(detectionRequests.length, 2);
  assert(detectionRequests.every((request) => request.url.includes('head_detection=true') && request.url.includes('imgsz=640')));
  console.log('UI checks passed: record filters, labels, hover, merge direction, selection, cancel, deletion, stale-view cleanup.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
