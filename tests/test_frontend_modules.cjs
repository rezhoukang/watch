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
  set innerHTML(value) { this.html = value; this.children = []; }
  get innerHTML() { return this.html || ''; }
  scrollIntoView() {}
  getBoundingClientRect() { return { left: 0 }; }
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
  const assetVersion = html.match(/app\.js\?v=(\d+)/)[1];
  await entry.link((specifier, parent) => {
    assert(specifier.endsWith(`?v=${assetVersion}`), `Module has stale asset version: ${specifier}`);
    return getModule(path.resolve(path.dirname(parent.identifier), specifier.split('?')[0]));
  });
  const styleVersion = html.match(/style\.css\?v=(\d+)/)[1];
  for (const match of fs.readFileSync(path.join(root, 'style.css'), 'utf8').matchAll(/@import url\('([^']+)'\)/g)) {
    assert(match[1].endsWith(`?v=${styleVersion}`), `Stylesheet has stale asset version: ${match[1]}`);
  }
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
  responses['/test-chart.json'] = { head_counts: [[100, 2], [150, 8], [200, 5]], fps: 25, start_frame: 100 };
  const player = nodes.get('videoResult').querySelector('video');
  player.readyState = 1; player.duration = 10;
  modules.get(path.join(root, 'js/charts.js')).namespace.showHeadChart('/test-chart.json');
  await flush();
  const chart = nodes.get('chartHost').children.at(-1);
  assert.equal(chart.querySelector('.headchart-stats').textContent, '最高人数：8 人 ｜ 最低人数：2 人 ｜ 全局平均人数：5.00 人');
  const chartScroll = chart.querySelector('.headchart-scroll');
  chartScroll.scrollLeft = 0;
  chartScroll.events.click({ clientX: 293 });
  assert.equal(player.currentTime, 2, 'Trimmed video seeks relative to start_frame');
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
    nodes.get(`${prefix}MergeIou`).value = '0.3';
    nodes.get(`${prefix}MergeConfidence`).value = '0.5';
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).mergeIou, 0.3);
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).mergeConfidence, 0.5);
    nodes.get(`${prefix}RtdetrBase`).checked = true;
    nodes.get(`${prefix}RtdetrBase`).events.change();
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).rtdetrBase, true);
    assert(nodes.get(`${prefix}ModelHint`).textContent.includes('RT-DETRv4 ≥ 0.25 为基准'));
    nodes.get(`${prefix}HeadDetection`).checked = false;
    nodes.get(`${prefix}HeadDetection`).events.change();
    assert.equal(nodes.get(`${prefix}Conf`).value, '0.7');
    assert.equal(nodes.get(`${prefix}Weights`).disabled, false);
    assert.equal(nodes.get(`${prefix}RtdetrBase`).disabled, true);
    assert.equal(nodes.get(`${prefix}MergeIou`).disabled, true);
    assert.equal(nodes.get(`${prefix}MergeConfidence`).disabled, true);
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).rtdetrBase, false);
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
  const keypoints = Array.from({ length: 17 }, () => [0, 0, 0]);
  for (const prefix of ['det', 'vid', 'ann']) {
    nodes.get(`${prefix}Weights`).value = 'yolov8m-pose.pt';
    nodes.get(`${prefix}Weights`).events.change();
    assert.equal(nodes.get(`${prefix}PoseOptions`).hidden, false);
    nodes.get(`${prefix}PoseRoi`).checked = true;
    nodes.get(`${prefix}RoiConfidence`).value = '0.7';
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).poseRoi, true);
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).roiConfidence, .7);
    for (const [direction, value] of [['Down', '0.1'], ['Up', '0.4']]) {
      nodes.get(`${prefix}Head${direction}RSlider`).value = value;
      nodes.get(`${prefix}Head${direction}RSlider`).events.input();
      assert.equal(nodes.get(`${prefix}Head${direction}R`).value, value);
      const field = direction === 'Down' ? 'headDownRThreshold' : 'headUpRThreshold';
      assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix)[field], Number(value));
      nodes.get(`${prefix}Head${direction}R`).value = value;
      nodes.get(`${prefix}Head${direction}R`).events.input();
      assert.equal(nodes.get(`${prefix}Head${direction}RSlider`).value, value);
    }
    nodes.get(`${prefix}HeadDownR`).value = '0.5';
    assert.throws(() => modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix), /低头 R 阈值必须小于抬头 R 阈值/);
    nodes.get(`${prefix}HeadDownR`).value = '0.1';
    nodes.get(`${prefix}Weights`).value = 'medium.pt';
    nodes.get(`${prefix}Weights`).events.change();
    assert.equal(nodes.get(`${prefix}PoseOptions`).hidden, true);
  }
  keypoints[5] = [20, 30, 0.9]; keypoints[7] = [40, 50, 0.8]; keypoints[9] = [70, 60, 0.2];
  keypoints[11] = [80, 70, 0.9];
  modules.get(path.join(root, 'js/viewer.js')).namespace.renderDetectResult({
    mode: 'pose', pose: { skeleton: [[5, 7], [7, 9]], keypoint_confidence: 0.5 },
    detections: [{ class_name: 'person', confidence: 0.9, bbox: [10, 10, 90, 70], keypoints,
      behavior: { state: '疑似阅读', roi: { polygon: [[20, 30], [60, 30], [40, 60]],
        objects: [{ class_name: 'book', confidence: .8, bbox: [30, 40, 45, 50] }] } } }],
  }, 'pose.png', 0.38, 0, 640);
  const poseLayer = viewer.querySelector('[data-role="boxes"]').children.find((node) => node.className === 'pose-layer');
  assert(poseLayer.innerHTML.includes('<line x1="20" y1="30" x2="40" y2="50"/>'));
  assert(!poseLayer.innerHTML.includes('cx="70"'));
  assert(!poseLayer.innerHTML.includes('cx="80"'));
  assert(poseLayer.innerHTML.includes('<polygon points="20,30 60,30 40,60"'));
  assert(poseLayer.innerHTML.includes('<rect x="30" y="40" width="15" height="10"'));
  assert(viewer.querySelector('[data-role="count"]').textContent.includes('疑似阅读 1 人'));
  assert(viewer.querySelector('[data-role="tablebody"]').children.some(row => row.innerHTML.includes('疑似阅读')));
  viewer.querySelector('[data-role="skeleton"]').events.change({ target: { checked: false } });
  assert(!viewer.querySelector('[data-role="boxes"]').children.some((node) => node.className === 'pose-layer'));
  assert.equal(nodes.has('globalBanner'), false);
  console.log('ES module startup, navigation and result viewer checks passed (14 modules, real HTML IDs).');
})().catch((error) => { console.error(error); process.exitCode = 1; });
