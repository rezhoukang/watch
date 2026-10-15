import { el, esc, api, banner } from './core.js?v=82';
import { showHeadChart } from './charts.js?v=82';
import { switchSubPane } from './navigation.js?v=82';
import { loadImages, detectImage } from './images.js?v=82';

async function loadOutputs() {
  const box = el('outputList');
  try {
    const data = await api('/api/videos');
    const outs = data.outputs || [];
    if (!outs.length) {
      box.innerHTML = '<div class="empty">暂无推理结果视频</div>';
      return;
    }
    box.innerHTML = '';
    outs.forEach((o) => {
      const item = document.createElement('div');
      item.className = 'item';
      item.innerHTML = `<span class="name">${esc(o.filename)}${o.head_detection ? ` · 目标检测融合（${o.parameters?.rtdetr_base ? 'RT-DETR' : 'medium'} 基准）` : ''}</span>`;

      const right = document.createElement('span');
      right.className = 'item-actions';

      // 预览：内嵌 <video>；折线图在独立面板里显示
      const play = document.createElement('button');
      play.className = 'btn small ghost';
      play.textContent = '预览';
      play.addEventListener('click', () => {
        const r = el('videoResult');
        const rtConf = o.parameters?.[o.parameters?.rtdetr_base ? 'base' : 'incoming']?.confidence ?? 0.2;
        r.innerHTML = (o.head_detection ? `<p>目标检测融合结果：${o.parameters?.rtdetr_base ? `RT-DETRv4 ${esc(rtConf)} 基准 + medium 0.45 补齐` : `medium 0.45 基准 + RT-DETRv4 ${esc(rtConf)} 补齐`}，640；合并 IoU ≤ ${esc(o.parameters?.merge_iou ?? 0)}；绿框为基准，橙框为补充。</p>` : '') + `<video src="${esc(o.url)}" controls autoplay></video>`;
        if (o.head_detection) {
          const gate = document.createElement('p');
          gate.textContent = `自定义 IoU 仅用于待补入框置信度 > ${o.parameters?.merge_confidence ?? 0}；其余仍要求 IoU=0。`;
          r.appendChild(gate);
        }
        showHeadChart(o.chart_url, o.filename);
      });

      // 图预览：该段「召回率最高帧」已提取到待推理图片，点它切过去并自动推理
      let pv = null;
      if (o.best_frame) {
        pv = document.createElement('button');
        pv.className = 'btn small ghost';
        pv.textContent = '图预览';
        pv.title = '把该段检出目标数最多的一帧提取到「待推理图片」，并自动推理';
        pv.addEventListener('click', async () => {
          // 切到图片推理 tab 的「待推理图片」子页
          document.querySelector('.tab[data-tab="detect"]').click();
          switchSubPane('images');
          el('detHeadDetection').checked = !!o.head_detection;
          el('detRtdetrBase').checked = !!o.parameters?.rtdetr_base;
          el('detMergeIou').value = String(o.parameters?.merge_iou ?? 0);
          el('detMergeConfidence').value = String(o.parameters?.merge_confidence ?? 0);
          el('detHeadDetection').dispatchEvent(new Event('change'));
          if (!o.head_detection && o.parameters?.weights) {
            el('detWeights').value = o.parameters.weights;
            el('detWeights').dispatchEvent(new Event('change'));
          }
          await loadImages();
          await detectImage(o.best_frame);
        });
        right.appendChild(pv);
      }

      // 删除
      const del = document.createElement('button');
      del.className = 'btn small danger';
      del.textContent = '删除';
      del.addEventListener('click', async () => {
        if (!confirm(`删除推理结果「${o.filename}」？`)) return;
        try {
          await api(`/api/outputs/${encodeURIComponent(o.filename)}`, { method: 'DELETE' }, 15000);
          await loadOutputs();
        } catch (e) {
          banner(`删除失败：${e instanceof Error ? e.message : String(e)}`);
        }
      });

      right.append(play, del);
      item.appendChild(right);
      box.appendChild(item);
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

el('refreshOutputs').addEventListener('click', loadOutputs);


export { loadOutputs };
