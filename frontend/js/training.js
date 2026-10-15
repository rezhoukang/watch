import { el, esc, api, showError } from './core.js?v=88';
import { DEFAULT_WEIGHTS, loadEnv } from './models.js?v=88';
import { pollJob } from './jobs.js?v=88';

/* ==================== 训练 ==================== */

async function loadDatasets() {
  const box = el('datasetList');
  const sel = el('trainDataset');
  const annSel = el('annotateDataset');
  const previous = [sel.value, annSel.value];
  try {
    const data = await api('/api/datasets');
    sel.innerHTML = '';
    annSel.innerHTML = '';
    if (!data.datasets.length) {
      box.innerHTML = '<div class="empty">暂无数据集</div>';
      sel.innerHTML = '<option value="">（无数据集）</option>';
      annSel.innerHTML = '<option value="">（无数据集）</option>';
      return;
    }
    box.innerHTML = '';
    data.datasets.forEach((d) => {
      const item = document.createElement('div');
      item.className = 'item';
      item.innerHTML =
        `<span class="name">${esc(d.name)}</span>` +
        `<span class="meta">${esc(d.images)} 张图片${d.has_yaml ? ' · 已有 yaml' : ''}</span>`;
      box.appendChild(item);

      for (const s of [sel, annSel]) {
        const o = document.createElement('option');
        o.value = d.name;
        o.textContent = d.name;
        s.appendChild(o);
      }
    });
    [sel, annSel].forEach((select, i) => {
      if (data.datasets.some(d => d.name === previous[i])) select.value = previous[i];
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

el('trainBtn').addEventListener('click', async () => {
  const box = el('trainResult');
  const dataset = el('trainDataset').value;

  if (!dataset) {
    box.innerHTML = '<span style="color:#c5221f">请先准备数据集（data/datasets/&lt;名称&gt;/）</span>';
    return;
  }

  const params = new URLSearchParams({
    dataset,
    weights: el('trainWeights').value || DEFAULT_WEIGHTS,
    epochs: el('trainEpochs').value,
    imgsz: el('trainImgsz').value,
    batch: el('trainBatch').value,
    lr0: el('trainLr0').value,
  });

  box.innerHTML = '正在启动训练…';
  const t0 = Date.now();

  try {
    const res = await api(`/api/train?${params}`, { method: 'POST' }, 30000);
    if (!res || !res.job_id) {
      throw new Error(`后端未返回 job_id：${JSON.stringify(res).slice(0, 200)}`);
    }

    pollJob(
      res.job_id,
      box,
      (job) => {
        let html = `<div class="summary">${esc(job.message)}</div>`;
        if (job.metrics && Object.keys(job.metrics).length) {
          html += '<table><tr><th>指标</th><th>值</th></tr>';
          Object.entries(job.metrics).forEach(([k, v]) => {
            html += `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`;
          });
          html += '</table>';
        }
        if (job.result_url) {
          html +=
            `<p style="margin-top:10px">最佳权重：` +
            `<a href="${esc(job.result_url)}" download>下载 best.pt</a></p>`;
        }
        box.innerHTML = html;
        loadEnv();
      },
      t0
    );
  } catch (err) {
    showError(box, err, '训练启动失败');
  }
});


export { loadDatasets };
