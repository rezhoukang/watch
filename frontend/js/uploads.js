import { el, api } from './core.js?v=69';
import { loadImages } from './images.js?v=69';
import { loadVideos } from './video.js?v=69';

/* ==================== 上传（图片 / 视频）==================== */

/**
 * kind: 'image' | 'video'
 */
function bindDropZone(zoneId, inputId, btnId, statusId, kind, uploadPath, onDone) {
  const zone = el(zoneId);
  const fileInput = el(inputId);
  const btn = el(btnId);
  const status = el(statusId);
  let files = [];

  const update = () => {
    const n = files.length;
    btn.disabled = n === 0;
    btn.textContent = n > 0 ? `上传 ${n} 个文件` : kind === 'image' ? '上传图片' : '上传视频';
    if (n > 0) {
      status.className = 'status';
      status.textContent = `已选择 ${n} 个文件`;
    }
  };

  fileInput.addEventListener('change', () => {
    files = Array.from(fileInput.files || []);
    update();
  });

  zone.addEventListener('click', () => fileInput.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const list = Array.from((e.dataTransfer && e.dataTransfer.files) || []);
    files = list.filter((f) => f.type.startsWith(`${kind}/`));
    update();
  });

  btn.addEventListener('click', async () => {
    if (!files.length) return;
    btn.disabled = true;
    status.className = 'status';
    status.textContent = '上传中…';

    const ok = [];
    const fail = [];

    try {
      for (const f of files) {
        const fd = new FormData();
        fd.append('file', f);
        try {
          const d = await api(uploadPath, { method: 'POST', body: fd }, 120000);
          ok.push(d && d.overwritten ? `${f.name}（已覆盖同名文件）` : f.name);
        } catch (e) {
          fail.push(`${f.name}: ${e instanceof Error ? e.message : String(e)}`);
        }
      }

      status.className = 'status' + (fail.length ? ' error' : '');
      status.textContent = `成功 ${ok.length} 个${fail.length ? `，失败 ${fail.length} 个：${fail.join('；')}` : ''}`;
    } finally {
      files = [];
      fileInput.value = '';
      update();
      if (onDone) onDone();
    }
  });
}

bindDropZone('dropZone', 'fileInput', 'uploadBtn', 'status', 'image', '/api/upload', loadImages);
bindDropZone('dropZoneVideo', 'videoInput', 'uploadVideoBtn', 'videoStatus', 'video', '/api/upload_video', loadVideos);
