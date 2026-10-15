import { esc, api, banner, showError } from './core.js?v=71';

/* ==================== 通用：轮询后台任务 ====================
 * - 单次查询 15 秒超时
 * - 偶发失败自动重试（连续 5 次才判失败）
 * - 一直显示已用时，避免"看起来卡死"
 */
function pollJob(jobId, box, onFinish, t0 = Date.now()) {
  let failCount = 0;

  /** 取消按钮（点击调取消接口，失败也提示） */
  const cancelBtn = () =>
    `<button class="btn small danger" data-cancel="${esc(jobId)}">取消任务</button>`;

  // 给 box 绑定取消事件（委托，避免反复重建）
  if (!box._cancelBound) {
    box._cancelBound = true;
    box.addEventListener('click', async (ev) => {
      const b = ev.target.closest('[data-cancel]');
      if (!b) return;
      b.disabled = true;
      b.textContent = '取消中…';
      try {
        await api(`/api/jobs/${encodeURIComponent(b.dataset.cancel)}/cancel`, { method: 'POST' }, 15000);
      } catch (e) {
        banner(`取消失败：${e instanceof Error ? e.message : String(e)}`);
        b.disabled = false;
        b.textContent = '取消任务';
      }
    });
  }

  const timer = setInterval(async () => {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    try {
      const job = await api(`/api/jobs/${jobId}`, {}, 15000);
      failCount = 0;

      const pct = job.total ? Math.round((job.progress / job.total) * 100) : 0;
      box.innerHTML =
        `<div>${esc(job.message)}</div>` +
        `<div class="progress"><div style="width:${pct}%"></div></div>` +
        `<div style="font-size:12px;color:#80868b">` +
        `${esc(job.progress)}/${esc(job.total || '?')}（${pct}%）` +
        `｜已用时 <span class="elapsed">${elapsed}s</span></div>` +
        ` <div style="margin-top:8px">${cancelBtn()}</div>`;

      if (job.status !== 'running') {
        clearInterval(timer);
        onFinish(job);
      }
    } catch (err) {
      failCount += 1;
      console.warn(`[pollJob] 第 ${failCount} 次状态查询失败`, err);
      if (failCount >= 5) {
        clearInterval(timer);
        showError(box, err, `任务状态查询连续失败 ${failCount} 次（任务 ID：${jobId}）`);
      } else {
        box.innerHTML =
          `<div>任务进行中…（状态查询临时失败 ${failCount}/5，自动重试中）</div>` +
          `<div style="font-size:12px;color:#80868b">已用时 <span class="elapsed">${elapsed}s</span></div>` +
          ` <div style="margin-top:8px">${cancelBtn()}</div>`;
      }
    }
  }, 1500);
}


export { pollJob };
