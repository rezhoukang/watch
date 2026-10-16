import { el } from './core.js?v=90';
import { hideHoverPreview } from './preview.js?v=90';

let loadHistory = () => {};

/* ==================== 标签页切换 ==================== */

function initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate }) {
  loadHistory = loadInferenceHistory;
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');

      const name = tab.dataset.tab || '';
      el(`tab-${name}`).classList.add('active');

      if (name !== 'detect') hideHoverPreview();

      if (name === 'detect') loadImages();
      if (name === 'headcut') loadHeadCuts();
      if (name === 'video') loadVideos();
      // 训练页只要数据集下拉框；数据集管理页还要标注缩略图
      if (name === 'train') loadDatasets();
      if (name === 'dataset') loadDatasets().then(loadAnnotateGrid);
      if (name === 'aggregate') loadAggregate();
    });
  });

  // 子标签只作用于自己所在的标签页：聚合操作有自己的子标签（switchAggregatePane）
  document.querySelectorAll('#tab-detect .subtab').forEach((t) => {
    t.addEventListener('click', () => switchSubPane(t.dataset.sub || 'images'));
  });
}

/* ==================== 图片推理：待推理图库 / 当前结果 / 历史结果图库 ==================== */

const SUBTAB_PANES = { images: 'detect-pane-images', result: 'detect-pane-result', history: 'detect-pane-history' };

/** pane: 'images' | 'result' | 'history' */
function switchSubPane(pane) {
  Object.entries(SUBTAB_PANES).forEach(([key, id]) => {
    el(id).classList.toggle('active', key === pane);
  });
  document.querySelectorAll('#tab-detect .subtab').forEach((t) => {
    t.classList.toggle('active', t.dataset.sub === pane);
  });
  el('refreshImages').hidden = pane !== 'images';
  if (pane !== 'images') hideHoverPreview();
  if (pane === 'history') loadHistory();
  window.scrollTo(0, 0);
}

export { initNavigation, switchSubPane };
