import { el } from './core.js?v=83';

/* ==================== 缩略图悬浮大图预览 ==================== */

let hoverEv = null;

/** 按鼠标位置摆放预览框；贴到右/下边缘时自动翻到另一侧 */
function placeHoverPreview(ev) {
  const box = el('hoverPreview');
  const pad = 18;
  const rect = box.getBoundingClientRect();

  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + rect.width > window.innerWidth - 10) x = ev.clientX - pad - rect.width;
  if (y + rect.height > window.innerHeight - 10) y = window.innerHeight - 10 - rect.height;
  if (x < 10) x = 10;
  if (y < 10) y = 10;

  box.style.left = `${Math.round(x)}px`;
  box.style.top = `${Math.round(y)}px`;
}

function showHoverPreview(src, name, ev, enlarge = false) {
  const box = el('hoverPreview');
  const img = box.querySelector('img');
  hoverEv = ev;
  if (img.getAttribute('src') !== src) img.src = src;
  el('hoverPreviewCap').textContent = name || '';
  box.classList.toggle('is-enlarged', enlarge);
  box.style.display = 'block';
  placeHoverPreview(ev);
}

function hideHoverPreview() {
  hoverEv = null;
  const box = document.getElementById('hoverPreview');
  if (box) box.style.display = 'none';
}

function bindImageHover(target, getPreview) {
  target.addEventListener('mouseenter', (ev) => {
    const preview = getPreview();
    if (preview?.src) showHoverPreview(preview.src, preview.name, ev, preview.enlarge);
  });
  target.addEventListener('mousemove', (ev) => {
    if (hoverEv) { hoverEv = ev; placeHoverPreview(ev); }
  });
  target.addEventListener('mouseleave', hideHoverPreview);
}

// 图片下载完才知道真实尺寸，加载完再定位一次
el('hoverPreview').querySelector('img').addEventListener('load', () => {
  if (hoverEv) placeHoverPreview(hoverEv);
});


export { bindImageHover, hideHoverPreview };
