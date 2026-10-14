// Run with: node --experimental-vm-modules tests/test_frontend_modules.cjs
// Load the real ES module graph with a minimal DOM and fake read-only API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor() {
    this.value = ''; this.dataset = {}; this.events = {}; this.children = []; this.queries = new Map();
    this.complete = true; this.naturalWidth = 100; this.naturalHeight = 80;
    this.style = { setProperty() {} }; this.classList = { add() {}, remove() {}, toggle() {} };
  }
  addEventListener(name, callback) { this.events[name] = callback; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  querySelector(selector) {
    if (!this.queries.has(selector)) this.queries.set(selector, new Element());
    return this.queries.get(selector);
  }
  querySelectorAll() { return []; }
  removeAttribute() {}
  setAttribute() {}
}
const root = path.resolve(__dirname, '../frontend');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const nodes = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => [match[1], new Element()]));
const tabs = ['detect', 'video', 'train', 'dataset'].map((name) => {
  const tab = new Element(); tab.dataset.tab = name; return tab;
});
const subtabs = ['images', 'result', 'history'].map((name) => {
  const tab = new Element(); tab.dataset.sub = name; return tab;
});
const calls = [];
const responses = {
  '/api/yolo/status': { message: 'ready', available: true, weights: ['test.pt'], default_weights: 'test.pt' },
  '/api/images': { images: [] }, '/api/videos': { videos: [], outputs: [] },
  '/api/inferences': { records: [] }, '/api/datasets': { datasets: [] },
};
const context = vm.createContext({
  console, setTimeout, clearTimeout, AbortController, URLSearchParams,
  window: { addEventListener() {}, scrollTo() {} },
  document: {
    getElementById: (id) => nodes.get(id) || null,
    createElement: () => new Element(), addEventListener() {},
    querySelectorAll: (selector) => selector === '.tab' ? tabs : selector === '.subtab' ? subtabs : [],
    body: new Element(),
  },
  fetch: async (url) => {
    assert(Object.hasOwn(responses, url), `Unexpected request: ${url}`);
    calls.push(url); return { ok: true, text: async () => JSON.stringify(responses[url]) };
  },
});
const modules = new Map();
function getModule(filename) {
  if (!modules.has(filename)) modules.set(filename, new vm.SourceTextModule(fs.readFileSync(filename, 'utf8'), {
    context, identifier: filename,
  }));
  return modules.get(filename);
}
const flush = () => new Promise((resolve) => setImmediate(resolve));
(async () => {
  const entry = getModule(path.join(root, 'app.js'));
  await entry.link((specifier, parent) => getModule(path.resolve(path.dirname(parent.identifier), specifier)));
  await entry.evaluate(); await flush();
  assert.equal(modules.size, 14);
  assert(calls.includes('/api/yolo/status'));
  assert(calls.includes('/api/images'));
  assert(calls.includes('/api/videos'));
  for (const tab of tabs) { await tab.events.click(); await flush(); }
  for (const tab of subtabs) { await tab.events.click(); await flush(); }
  assert(calls.includes('/api/datasets'));
  assert(calls.includes('/api/inferences'));
  assert.equal(nodes.get('envStatus').className, 'badge ok');
  nodes.get('detConf').value = '0.38'; nodes.get('detImgsz').value = '960';
  nodes.get('detHeadDetection').checked = true;
  nodes.get('detHeadDetection').events.change();
  assert.equal(nodes.get('detConf').value, '0.45');
  assert.equal(nodes.get('detImgsz').value, '640');
  assert.equal(nodes.get('detWeights').value, 'medium.pt');
  assert.equal(nodes.get('detImgsz').disabled, true);
  nodes.get('detHeadDetection').checked = false;
  nodes.get('detHeadDetection').events.change();
  assert.equal(nodes.get('detConf').value, '0.38');
  assert.equal(nodes.get('detImgsz').value, '960');
  assert.equal(nodes.get('detImgsz').disabled, false);
  for (const prefix of ['vid', 'ann']) {
    nodes.get(`${prefix}Conf`).value = '0.7';
    nodes.get(`${prefix}HeadDetection`).checked = true;
    nodes.get(`${prefix}HeadDetection`).events.change();
    assert.equal(nodes.get(`${prefix}Conf`).value, '0.45');
    assert.equal(nodes.get(`${prefix}Weights`).disabled, true);
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).headDetection, true);
    nodes.get(`${prefix}HeadDetection`).checked = false;
    nodes.get(`${prefix}HeadDetection`).events.change();
    assert.equal(nodes.get(`${prefix}Conf`).value, '0.7');
    assert.equal(nodes.get(`${prefix}Weights`).disabled, false);
  }
  let opened, deleted;
  modules.get(path.join(root, 'js/viewer.js')).namespace.renderDetectResult({
    record_id: 'example', source_url: '/original.png', mode: 'rtdetrv4',
    detections: [{ class_name: 'head', confidence: 0.8, bbox: [10, 20, 30, 40] }],
  }, 'example.png', 0.38, 0, 640, {
    onRecordOpen: (id) => { opened = id; }, onRecordDelete: (id) => { deleted = id; },
  });
  const viewer = nodes.get('detectResult');
  assert.equal(opened, 'example');
  assert.equal(viewer.querySelector('[data-role="boxes"]').children.length, 1);
  assert(viewer.querySelector('[data-role="count"]').textContent.includes('1 人'));
  assert.equal(viewer.querySelector('[data-role="viewer"]').events.mouseenter, undefined);
  viewer.querySelector('[data-role="delete-record"]').events.click();
  assert.equal(deleted, 'example');
  assert.equal(nodes.has('globalBanner'), false);
  console.log('ES module startup, navigation and result viewer checks passed (14 modules, real HTML IDs).');
})().catch((error) => { console.error(error); process.exitCode = 1; });
