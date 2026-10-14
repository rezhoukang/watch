
/* Shared DOM utilities, error reporting and HTTP requests. */

/* ==================== 通用工具 ==================== */

/** 取元素，取不到就抛明确错误（避免"点了没反应"） */
function el(id) {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`页面元素缺失：#${id}（请强制刷新页面 Ctrl+F5）`);
  }
  return node;
}

/** HTML 转义，避免内容破坏页面结构 */
function esc(v) {
  return String(v == null ? '' : v).replace(/[&<>"']/g, (c) => {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return map[c];
  });
}

/** 把错误显示到指定区域（同时打印完整堆栈到控制台） */
function showError(box, err, prefix = '失败') {
  const e = err instanceof Error ? err : new Error(String(err));
  console.error(`[${prefix}]`, e);
  if (box) {
    box.innerHTML =
      `<div style="color:#c5221f;font-weight:600">${esc(prefix)}：${esc(e.message)}</div>` +
      `<div style="font-size:12px;color:#80868b;margin-top:6px">` +
      `错误类型：${esc(e.name)}｜详细堆栈见浏览器控制台（F12 → Console）</div>`;
  }
}

/** 顶部红条：捕获脚本级异常，避免静默失败 */
function banner(msg) {
  let node = document.getElementById('globalBanner');
  if (!node) {
    node = document.createElement('div');
    node.id = 'globalBanner';
    node.style.cssText =
      'position:fixed;top:0;left:0;right:0;z-index:99;background:#c5221f;color:#fff;' +
      'padding:8px 14px;font-size:13px;font-family:inherit';
    document.body.appendChild(node);
  }
  node.textContent = msg;
}

window.addEventListener('error', (ev) => {
  banner(`脚本异常：${ev.message}（${ev.filename}:${ev.lineno}）`);
});

window.addEventListener('unhandledrejection', (ev) => {
  const r = ev.reason;
  banner(`异步异常：${r instanceof Error ? r.message : String(r)}`);
});

/* 禁用浏览器对图片的「原生拖拽」
 *
 * 标注是「长按拖拽画框」的手势，而浏览器在按住图片拖动时会启动
 * 自己的拖拽（图片幽灵跟着光标、松手后可能触发导航），两者直接冲突。
 * 所以：
 *   1. 全局拦 dragstart（CSS 的 -webkit-user-drag 只管 Chromium，
 *      Firefox 不认，必须在 JS 里拦才跨浏览器）
 *   2. 标注器里的图片再设 draggable=false（掐掉可能发起拖拽的源头）
 */
document.addEventListener('dragstart', (ev) => {
  const t = ev.target;
  if (!t) return;
  // 图片本身，或标注器里的任何元素，都不允许发起原生拖拽
  if (t.tagName === 'IMG' || (t.closest && t.closest('#annotatorMask'))) {
    ev.preventDefault();
  }
});
el('annBaseImg').draggable = false;

/* ==================== 请求封装（带超时 + 明确错误）==================== */

async function api(path, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const t0 = Date.now();

  try {
    const res = await fetch(path, { ...options, signal: controller.signal });
    const text = await res.text();

    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (e) {
        data = null;
      }
    }

    if (!res.ok) {
      const detail = (data && (data.detail || data.message)) || text.slice(0, 200) || '(空响应体)';
      throw new Error(`HTTP ${res.status} ${res.statusText} — ${detail}`);
    }
    if (data === null) {
      throw new Error(`响应不是合法 JSON（HTTP ${res.status}）：${text.slice(0, 200)}`);
    }
    return data;
  } catch (err) {
    // 超时
    if (err && err.name === 'AbortError') {
      throw new Error(`请求超时：${timeoutMs / 1000} 秒内无响应 — ${path}`);
    }
    // 连不上后端
    if (err instanceof TypeError) {
      throw new Error(`无法连接后端（服务未启动 / 已关闭 / 端口不对？）：${err.message}`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
    console.debug(`[api] ${path} 用时 ${Date.now() - t0}ms`);
  }
}


export { el, esc, showError, banner, api };
