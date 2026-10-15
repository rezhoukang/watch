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
    this.style = { setProperty() {} };
    const classes = new Set();
    this.classList = {
      add: (...names) => names.forEach(name => classes.add(name)),
      remove: (...names) => names.forEach(name => classes.delete(name)),
      contains: name => classes.has(name),
      toggle: (name, force) => {
        const enabled = force === undefined ? !classes.has(name) : !!force;
        if (enabled) classes.add(name); else classes.delete(name);
        return enabled;
      },
    };
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
const tabs = ['detect', 'headcut', 'video', 'train', 'dataset', 'aggregate'].map((name) => {
  const tab = new Element(); tab.dataset.tab = name; return tab;
});
const subtabs = ['images', 'result', 'history'].map((name) => {
  const tab = new Element(); tab.dataset.sub = name; return tab;
});
const aggregateSubtabs = ['images', 'result'].map((name) => {
  const tab = new Element(); tab.dataset.agg = name; return tab;
});
const calls = [];
const responses = {
  '/api/yolo/status': { message: 'ready', available: true, weights: ['test.pt'], default_weights: 'test.pt' },
  '/api/images': { images: [] }, '/api/videos': { videos: [], outputs: [] },
  '/api/inferences': { records: [] }, '/api/datasets': { datasets: [] },
  '/api/head-cuts': { runs: [] },
};
const context = vm.createContext({
  console, setTimeout, clearTimeout, AbortController, URLSearchParams,
  // setInterval 只记录不真的跑：进度刷新用它，真的跑起来进程会一直挂着不退出
  setInterval: () => 1, clearInterval: () => {},
  window: { addEventListener() {}, scrollTo() {} },
  document: {
    getElementById: (id) => nodes.get(id) || null,
    createElement: () => new Element(), addEventListener() {},
    querySelectorAll: (selector) => selector === '.tab' ? tabs
      : selector === '#tab-detect .subtab' ? subtabs
      : selector === '.subtab[data-agg]' ? aggregateSubtabs : [],
    body: new Element(),
  },
  fetch: async (url, options = {}) => {
    assert(Object.hasOwn(responses, url), `Unexpected request: ${url}`);
    calls.push(url); return { ok: true, text: async () => JSON.stringify(typeof responses[url] === 'function' ? responses[url](options) : responses[url]) };
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
  assert.equal(modules.size, 16);
  assert(calls.includes('/api/yolo/status'));
  assert(calls.includes('/api/images'));
  assert(calls.includes('/api/videos'));
  for (const tab of tabs) { await tab.events.click(); await flush(); }
  for (const tab of subtabs) { await tab.events.click(); await flush(); }
  assert.equal(aggregateSubtabs.length, 2);
  for (const tab of aggregateSubtabs) { await tab.events.click(); await flush(); }
  assert(calls.includes('/api/datasets'));
  assert(calls.includes('/api/inferences'));
  assert(calls.includes('/api/head-cuts'));
  const cutRunId = 'f'.repeat(32);
  let cutStarted = false, cutPayload;
  responses['/api/images'] = { images: [{ filename: '课堂A.png', url: '/images/a.png' }] };
  responses['/api/head-cuts'] = options => {
    if (options.method === 'POST') { cutPayload = JSON.parse(options.body); cutStarted = true; return { run_id: cutRunId }; }
    return { runs: cutStarted ? [{ run_id: cutRunId, created_at: '2026-10-15T00:00:00Z', crop_count: 1 }] : [] };
  };
  responses[`/api/head-cuts/${cutRunId}`] = {
    run_id: cutRunId, status: 'done', progress: 1, total: 1, crop_count: 1, message: '完成', errors: [],
    inputs: [{ filename: '课堂A.png' }], images: [{ source: '课堂A.png', crop_count: 1, annotated_file: 'annotated/001.png' }],
    crops: [{ crop_id: 'HC001-001', source: '课堂A.png', class_name: 'head', confidence: .9,
      crop_bbox: [1, 2, 3, 4], raw_file: 'crops/HC001-001.png', marked_file: 'marked/HC001-001.png', mark: '' }],
  };
  await modules.get(path.join(root, 'js/head-cuts.js')).namespace.loadHeadCuts();
  nodes.get('cutSelectAll').checked = true;
  nodes.get('cutSelectAll').events.change({ target: nodes.get('cutSelectAll') });
  assert.equal(nodes.get('cutInfer').disabled, false);
  await nodes.get('cutInfer').events.click();
  assert.deepEqual(cutPayload, { filenames: ['课堂A.png'], rtdetr_base: false, merge_iou: .1, merge_confidence: .55 });
  assert.equal(nodes.get('cutHeadGrid').children.length, 1);
  assert(nodes.get('cutHeadGrid').children[0].innerHTML.includes('HC001-001'));
  // 切割完成后「1:N 识别」就可点；未做过人脸检测时点击会自动先检测再识别
  assert.equal(nodes.get('cutRecognizeFaces').disabled, false, '切割完成后识别按钮应可点');
  assert(nodes.get('cutRecognizeFaces').title.includes('先自动检测'), '未检测时应提示会自动先检测');
  let autoFaceCalls = 0, autoFacePayload, autoRecPayload;
  const recordRef = responses[`/api/head-cuts/${cutRunId}`];
  const originalGet = responses[`/api/head-cuts/${cutRunId}`];
  responses[`/api/head-cuts/${cutRunId}/faces`] = options => {
    autoFaceCalls++;
    autoFacePayload = JSON.parse(options.body);
    recordRef.face_detection = { job_id: 'auto-face', status: 'running', progress: 0, total: 1, config: { confidence: .5 }, message: '检测中' };
    return { job_id: 'auto-face', run_id: cutRunId };
  };
  // GET：检测任务从 running 一次性翻转到 done，并让裁剪带上已检出人脸
  responses[`/api/head-cuts/${cutRunId}`] = options => {
    if (options.method) return originalGet(options);
    if (recordRef.face_detection?.status === 'running') {
      recordRef.face_detection = { job_id: 'auto-face', status: 'done', progress: 1, total: 1, config: { confidence: .5 }, message: '人脸检测完成' };
      recordRef.crops[0].face_detection = { status: 'detected', face_count: 1,
        faces: [{ face_id: 'HC001-001_F01', confidence: .8, file: 'faces/HC001-001_F01.png' }] };
    }
    return recordRef;
  };
  responses[`/api/head-cuts/${cutRunId}/recognize`] = options => {
    autoRecPayload = JSON.parse(options.body);
    recordRef.recognition = { job_id: 'auto-rec', status: 'done', progress: 1, total: 1, config: { threshold: .4 }, message: '识别完成' };
    recordRef.crops[0].recognition = { crop_id: 'HC001-001', status: 'done', name: '张三',
      faces: [{ face_id: 'HC001-001_F01', status: 'matched', face_pixels: 26,
        match: { matched: true, entry_id: '0001', name: '张三', similarity: .95, photo_url: '/api/face-library/0001/photo' } }] };
    return { job_id: 'auto-rec', run_id: cutRunId };
  };
  await nodes.get('cutRecognizeFaces').events.click();
  await flush();
  assert.equal(autoFaceCalls, 1, '未检测时应自动触发一次批量检测');
  assert.deepEqual(autoFacePayload, { confidence: .5 });
  assert.deepEqual(autoRecPayload, { threshold: .4, min_confidence: .5, min_size: 24 });
  assert(nodes.get('cutFaceProgress').textContent.includes('识别完成'), '自动检测后应继续识别');
  assert(nodes.get('cutHeadGrid').children[0].children.some(node => node.innerHTML.includes('rec-hit')));
  // 恢复原始 mock 并清掉自动用例写入的检测/识别状态，避免污染后续用例
  responses[`/api/head-cuts/${cutRunId}`] = originalGet;
  delete recordRef.face_detection; delete recordRef.recognition;
  delete recordRef.crops[0].face_detection; delete recordRef.crops[0].recognition;
  assert(nodes.get('cutHeadGrid').children[0].children.every(child => child.type !== 'text' && !child.children.some(input => input.type === 'text')),
    '头管理卡片不应包含人工备注输入框');
  let facePayload;
  responses[`/api/head-cuts/${cutRunId}/faces`] = options => {
    facePayload = JSON.parse(options.body);
    const record = responses[`/api/head-cuts/${cutRunId}`];
    record.face_detection = { job_id: 'face-job', status: 'done', progress: 1, total: 1, config: { confidence: .5 }, message: '人脸检测完成' };
    record.crops[0].face_detection = { status: 'detected', face_count: 1, annotated_file: 'face_annotations/HC001-001.png',
      faces: [{ face_id: 'HC001-001_F01', confidence: .8, file: 'faces/HC001-001_F01.png' }] };
    return { job_id: 'face-job', run_id: cutRunId };
  };
  await nodes.get('cutDetectFaces').events.click();
  assert.deepEqual(facePayload, { confidence: .5 });
  assert(nodes.get('cutFaceProgress').textContent.includes('人脸检测完成'));
  const faceStrip = nodes.get('cutHeadGrid').children[0].children.find(node => node.className === 'head-face-strip');
  assert(faceStrip.innerHTML.includes('HC001-001_F01') && !faceStrip.innerHTML.includes('<img'),
    '人脸结果保留文字信息，不重复显示照片');
  nodes.get('cutFaceFilter').value = 'no_face'; nodes.get('cutFaceFilter').events.change();
  assert.equal(nodes.get('cutHeadGrid').children.length, 0);
  nodes.get('cutFaceFilter').value = ''; nodes.get('cutFaceFilter').events.change();
  assert.equal(nodes.get('cutHeadGrid').children.length, 1);
  // 1:N 识别：命中显示姓名与相似度，质量不足与未识别分别给出可追溯的徽标
  assert.equal(nodes.get('cutRecognizeFaces').disabled, false, '检出人脸后应可识别');
  let recPayload;
  responses[`/api/head-cuts/${cutRunId}/recognize`] = options => {
    recPayload = JSON.parse(options.body);
    const record = responses[`/api/head-cuts/${cutRunId}`];
    record.recognition = { job_id: 'rec-job', status: 'done', progress: 1, total: 1, config: { threshold: .4 }, message: '识别完成' };
    record.crops[0].recognition = { crop_id: 'HC001-001', status: 'done', name: '张三',
      faces: [{ face_id: 'HC001-001_F01', status: 'matched', face_pixels: 26,
        match: { matched: true, entry_id: '0001', name: '张三', similarity: .95,
          photo_url: '/api/face-library/0001/photo' } }] };
    return { job_id: 'rec-job', run_id: cutRunId };
  };
  await nodes.get('cutRecognizeFaces').events.click();
  assert.deepEqual(recPayload, { threshold: .4, min_confidence: .5, min_size: 24 });
  // 质量门槛可调：不再硬编码 0.8，界面改值后原样提交
  nodes.get('cutRecThreshold').value = '0.35';
  nodes.get('cutRecConfidence').value = '0.55';
  nodes.get('cutRecSize').value = '12';
  await nodes.get('cutRecognizeFaces').events.click();
  assert.deepEqual(recPayload, { threshold: .35, min_confidence: .55, min_size: 12 });
  nodes.get('cutRecThreshold').value = '0.4';
  nodes.get('cutRecConfidence').value = '0.5';
  nodes.get('cutRecSize').value = '24';
  assert(nodes.get('cutFaceProgress').textContent.includes('识别完成'));
  const recFigure = nodes.get('cutHeadGrid').children[0];
  assert(recFigure.children.some(node => node.innerHTML.includes('rec-hit') && node.innerHTML.includes('张三')));
  // 命中的要把人像库登记照放在右侧
  const hitHtml = recFigure.children.map(node => node.innerHTML).join('');
  assert(hitHtml.includes('rec-lib'), '命中项应包含库内登记照');
  assert(hitHtml.includes('/api/face-library/0001/photo'), '登记照应指向人脸库照片接口');
  assert(hitHtml.indexOf('rec-query') < hitHtml.indexOf('rec-lib'), '登记照应在查询脸右侧');
  nodes.get('cutRecFilter').value = 'matched'; nodes.get('cutRecFilter').events.change();
  assert.equal(nodes.get('cutHeadGrid').children.length, 1);
  nodes.get('cutRecFilter').value = 'unknown'; nodes.get('cutRecFilter').events.change();
  assert.equal(nodes.get('cutHeadGrid').children.length, 0, '未识别筛选应排除已命中的裁剪');
  // 质量不足：展示原因，且不计为已识别
  responses[`/api/head-cuts/${cutRunId}`].crops[0].recognition = { crop_id: 'HC001-001', status: 'done', name: '',
    faces: [{ face_id: 'HC001-001_F01', status: 'rejected', face_pixels: 12, reasons: ['原始人脸尺寸过小'], faces: [] }] };
  await modules.get(path.join(root, 'js/head-cuts.js')).namespace.loadHeadCuts();
  await flush();
  nodes.get('cutRecFilter').value = 'unknown'; nodes.get('cutRecFilter').events.change();
  assert.equal(nodes.get('cutHeadGrid').children.length, 1, '未识别筛选应包含质量不足的裁剪');
  assert(nodes.get('cutHeadGrid').children[0].children.some(node => node.innerHTML.includes('rec-skip')));
  nodes.get('cutRecFilter').value = ''; nodes.get('cutRecFilter').events.change();
  // 顺延到备选：标注是第几个候选，并在提示里说明被谁抢走，同时仍显示对应登记照
  responses[`/api/head-cuts/${cutRunId}`].crops[0].recognition = { crop_id: 'HC001-001', status: 'done', name: '王五',
    faces: [{ face_id: 'HC001-001_F01', status: 'matched', face_pixels: 26,
      match: { matched: true, entry_id: '0005', name: '王五', similarity: .45, rank: 2,
        photo_url: '/api/face-library/0005/photo' },
      conflict: { skipped: [{ entry_id: '0001', name: '张三', similarity: .55, rank: 1, taken_by: 'HC001-009_F01' }] } }] };
  await modules.get(path.join(root, 'js/head-cuts.js')).namespace.loadHeadCuts();
  await flush();
  const rankHtml = nodes.get('cutHeadGrid').children[0].children.map(node => node.innerHTML).join('');
  assert(rankHtml.includes('rec-rank'), '顺延到备选应标注是第几个候选');
  assert(rankHtml.includes('备选#2'), '应显示备选序号');
  assert(rankHtml.includes('HC001-009_F01'), '提示里应说明被哪张脸抢走');
  assert(rankHtml.includes('/api/face-library/0005/photo'), '备选命中也要显示对应登记照');
  // 未识别：候选被别的脸占用时给出可追溯的原因
  responses[`/api/head-cuts/${cutRunId}`].crops[0].recognition = { crop_id: 'HC001-001', status: 'done', name: '',
    faces: [{ face_id: 'HC001-001_F01', status: 'unknown', face_pixels: 26,
      match: { matched: false, entry_id: '0001', name: '张三', similarity: .55, reason: 'taken' },
      conflict: { skipped: [{ entry_id: '0001', name: '张三', similarity: .55, rank: 1, taken_by: 'HC001-009_F01' }] } }] };
  await modules.get(path.join(root, 'js/head-cuts.js')).namespace.loadHeadCuts();
  await flush();
  const takenHtml = nodes.get('cutHeadGrid').children[0].children.map(node => node.innerHTML).join('');
  assert(takenHtml.includes('未识别'), '被抢占后没有可用备选应判为未识别');
  assert(takenHtml.includes('已被得分更高的另一张脸占用'), '未识别要说明是候选被抢占');
  nodes.get('cutRecFilter').value = ''; nodes.get('cutRecFilter').events.change();
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
    const originalConfidence = nodes.get(`${prefix}Conf`).value;
    nodes.get(`${prefix}Weights`).value = 'yolov8m-pose.pt';
    nodes.get(`${prefix}Weights`).events.change();
    assert.equal(nodes.get(`${prefix}PoseOptions`).hidden, false);
    assert.equal(nodes.get(`${prefix}Conf`).value, '0.6');
    nodes.get(`${prefix}RoiConfidence`).value = '';
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).roiConfidence, .03);
    nodes.get(`${prefix}Conf`).value = '0.78';
    nodes.get(`${prefix}MergeIou`).events.input();
    assert.equal(nodes.get(`${prefix}Conf`).value, '0.78', '更新其他控件不能重置手动置信度');
    nodes.get(`${prefix}HeadDetection`).checked = true;
    nodes.get(`${prefix}HeadDetection`).events.change();
    assert.equal(nodes.get(`${prefix}Conf`).value, '0.45');
    nodes.get(`${prefix}HeadDetection`).checked = false;
    nodes.get(`${prefix}HeadDetection`).events.change();
    assert.equal(nodes.get(`${prefix}Conf`).value, '0.78');
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
    assert.equal(nodes.get(`${prefix}Conf`).value, originalConfidence);
    nodes.get(`${prefix}Weights`).value = 'yolov8m-pose.pt';
    nodes.get(`${prefix}Weights`).events.change();
    assert.equal(nodes.get(`${prefix}Conf`).value, '0.78', '再次选择 Pose 保留手动设置');
    nodes.get(`${prefix}Conf`).value = '';
    assert.equal(modules.get(path.join(root, 'js/models.js')).namespace.readDetectOptions(prefix).conf, '0.6');
    nodes.get(`${prefix}Conf`).value = '0.78';
    nodes.get(`${prefix}Weights`).value = 'medium.pt';
    nodes.get(`${prefix}Weights`).events.change();
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
  // 聚合操作：点图片 → 目标检测 / 人脸识别（姓名贴到目标） / 姿势检测与绑定
  let aggregatePayload;
  const aggregateUrl = `/api/aggregate/image/${encodeURIComponent('课堂A.png')}`;
  responses[aggregateUrl] = options => {
    aggregatePayload = JSON.parse(options.body);
    return {
      source: '课堂A.png', result: '课堂A_agg.png', result_url: '/api/aggregate/results/%E8%AF%BE%E5%A0%82A_agg.png',
      count: 1, elapsed_seconds: 4.2,
      steps: {
        detect: { enabled: true, mode: 'head_fusion', count: 1, rule: 'medium.pt 0.45 为基准 + best_stg2.safetensors 0.25 补齐', elapsed_seconds: 1.1 },
        face: { enabled: true, face_count: 1, matched: 1, threshold: 0.4, library_count: 155, elapsed_seconds: 1.6 },
        pose: { enabled: true, count: 1, bound: 1, unbound: 0, min_iou: 0, elapsed_seconds: 1.5 },
      },
      bind: { rule: '链式贪心 + 一一对应：只保留 IoU > 0 的配对', face_rule: '链式贪心 + 一一对应', min_iou: 0 },
      image_width: 200, image_height: 400,
      summary: { targets: 1, named_targets: 1, faces: 1, matched_faces: 1, unknown_faces: 0, rejected_faces: 0, poses: 1, bound_poses: 1 },
      unbound_faces: [], unbound_poses: [],
      targets: [{ class_name: 'head', confidence: 0.91, bbox: [10, 10, 90, 200], name: '张三', entry_id: '0001', similarity: 0.62,
        face_id: 'F01', poses: [{ behavior: { state: '低头' }, bind: { iou: 0.12, distance: 3.5 } }] }],
    };
  };
  assert.equal(calls.includes(aggregateUrl), false);
  await modules.get(path.join(root, 'js/aggregate.js')).namespace.loadAggregate();
  await flush();
  assert.equal(nodes.get('aggImageGrid').children.length, 1);
  nodes.get('aggFaceRecognition').checked = true;
  nodes.get('aggPose').checked = true;
  await nodes.get('aggImageGrid').children[0].events.click();
  await flush();
  assert.deepEqual(aggregatePayload, {
    face: true, pose: true, threshold: 0.4, face_confidence: 0.5, pose_confidence: 0.6,
  });
  assert(nodes.get('aggStatus').textContent.includes('贴姓名 1 个'));
  const aggBox = nodes.get('aggResult');
  assert(aggBox.innerHTML.includes('data-role="viewer"') && aggBox.innerHTML.includes('class="boxes"'), '结果要和图片推理一样叠框渲染');
  assert(aggBox.innerHTML.includes('data-role="aggfilter"'), '要有与图片推理同款的胶囊筛选');
  assert(aggBox.innerHTML.includes('张三') && aggBox.innerHTML.includes('agg-name'), '姓名和相似度要写进明细表');
  assert(aggBox.innerHTML.includes('低头'), '绑定上的姿势要写进明细表');
  assert(aggBox.innerHTML.includes('medium.pt 0.45 为基准'), '结果里要说明目标检测用的哪套规则');
  assert(aggBox.innerHTML.includes('/api/aggregate/results/'), '后端渲染图仍要附对照链接');
  // 只画 head 框，且框上挂着姓名 + 姿势标签（DOM 桩不解析 HTML，直接查叠框层）
  const headLayer = aggBox.querySelector('[data-role="boxes"]');
  assert.equal(headLayer.children.length, 1);
  assert.equal(headLayer.children[0].style.left, '5%');
  assert.equal(headLayer.children[0].style.top, '2.5%');
  assert.equal(headLayer.children[0].style.width, '40%');
  assert.equal(headLayer.children[0].children[0].textContent, '张三 0.62 · 低头');
  assert.equal(aggBox.querySelector('#aggResultCount').textContent.includes('图上 1 个 head 框'), true);
  const aggViewer = aggBox.querySelector('[data-role="viewer"]');
  const tagToggle = aggBox.querySelector('[data-role="alwaystag"]');
  assert.equal(tagToggle.checked, false);
  assert.equal(aggViewer.classList.contains('always-tags'), false);
  const beforeTagToggle = calls.length;
  tagToggle.checked = true;
  tagToggle.events.change({ target: tagToggle });
  assert.equal(aggViewer.classList.contains('always-tags'), true);
  aggBox.querySelector('[data-role="baseimg"]').events.load();
  assert.equal(aggViewer.classList.contains('always-tags'), true, '重绘保留常显标签');
  assert.equal(calls.length, beforeTagToggle, '切换标签显示不能触发推理请求');
  // 勾掉两步后只跑目标检测，参数随勾选变化
  nodes.get('aggFaceRecognition').checked = false;
  nodes.get('aggPose').checked = false;
  nodes.get('aggRecThreshold').value = '0.35';
  nodes.get('aggPoseConfidence').value = '';
  await nodes.get('aggImageGrid').children[0].events.click();
  await flush();
  assert.equal(aggregatePayload.face, false);
  assert.equal(aggregatePayload.pose, false);
  assert.equal(aggregatePayload.threshold, 0.35);
  assert.equal(aggregatePayload.pose_confidence, 0.6, '清空输入框要回落到默认置信度');
  assert.equal(aggBox.querySelector('[data-role="alwaystag"]').checked, true, '切换图片后保留常显标签选择');
  assert.equal(aggViewer.classList.contains('always-tags'), true);
  tagToggle.checked = false;
  tagToggle.events.change({ target: tagToggle });
  assert.equal(aggViewer.classList.contains('always-tags'), false);
  // 点图后要自动切到「聚合结果」子页，并把状态切回图片页的刷新按钮藏起来
  assert.equal(aggregateSubtabs[0].dataset.agg, 'images');
  assert.equal(nodes.get('aggRefresh').hidden, true);
  nodes.get('aggRecThreshold').value = '2';
  await nodes.get('aggImageGrid').children[0].events.click();
  await flush();
  assert(aggBox.innerHTML.includes('识别阈值必须在 0 到 1 之间'), '参数越界要报错而不是静默失败');
  assert.equal(nodes.has('globalBanner'), false);
  console.log('ES module startup, navigation and result viewer checks passed (16 modules, real HTML IDs).');
})().catch((error) => { console.error(error); process.exitCode = 1; });
