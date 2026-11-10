import { el, api } from './core.js?v=95';
import { loadImages } from './images.js?v=127';
import { loadVideos } from './video.js?v=95';

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
  zone.addEventListener('drop', async (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const entries = kind === 'image' ? [...(e.dataTransfer?.items || [])].map(item => item.webkitGetAsEntry?.()).filter(Boolean) : [];
    if (entries.some(entry => entry.isDirectory)) {
      if (folderUploadBusy) return;
      folderUploadBusy = true;
      el('uploadFolderBtn').disabled = true;
      el('folderUploadStatus').textContent = '正在扫描拖入的文件夹…';
      try {
        await uploadFolderImages(await collectDroppedImages(entries));
      } catch (error) {
        el('folderUploadStatus').textContent = `扫描文件夹失败：${error.message}`;
      } finally {
        folderUploadBusy = false;
        el('uploadFolderBtn').disabled = false;
      }
      return;
    }
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


let folderUploadBusy = false;
const supportedImage = name => /\.(jpg|jpeg|png|bmp|webp)$/i.test(name);

async function collectDroppedImages(entries) {
  const images = [];
  async function visit(entry, parent) {
    if (entry.isDirectory) {
      const path = parent ? `${parent}/${entry.name}` : entry.name;
      const reader = entry.createReader();
      // Browsers return directory contents in pages, often 100 entries at a time.
      let children;
      do {
        children = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        for (const child of children) await visit(child, path);
      } while (children.length);
    } else if (entry.isFile && supportedImage(entry.name)) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      images.push({ file, relativePath: parent ? `${parent}/${entry.name}` : '' });
    }
  }
  for (const entry of entries) await visit(entry, '');
  return images;
}

async function uploadFolderImages(images) {
  const status = el('folderUploadStatus');
  let completed = 0, failed = 0;
  const failures = [];
  for (const {file, relativePath} of images) {
    status.textContent = `上传 ${completed + failed + 1}/${images.length} · ${relativePath || file.name}`;
    const body = new FormData();
    body.append('file', file);
    if (relativePath) body.append('relative_path', relativePath);
    try {
      await api(relativePath ? '/api/upload-folder-image' : '/api/upload', { method: 'POST', body }, 120000);
      completed++;
    } catch (error) {
      failed++;
      if (relativePath && /^HTTP (404|405)\b/.test(error.message)) {
        status.textContent = `文件夹上传已停止，成功 ${completed} 张，失败 ${failed} 张，未上传 ${images.length - completed - failed} 张。后端尚未加载文件夹上传接口，请重启项目服务后重新选择文件夹上传。`;
        if (completed) await loadImages();
        return;
      }
      failures.push(`${relativePath || file.name}：${error.message}`);
    }
  }
  status.textContent = images.length
    ? `文件夹上传完成，成功 ${completed} 张，失败 ${failed} 张；其余文件类型已跳过。${failures.join('；')}`
    : '所选文件夹中没有支持的图片。';
  await loadImages();
}

el('uploadFolderBtn').addEventListener('click', () => el('uploadFolderInput').click());
el('uploadFolderInput').addEventListener('change', async event => {
  if (folderUploadBusy) { event.target.value = ''; return; }
  const images = [...event.target.files].filter(file => supportedImage(file.name))
    .map(file => ({ file, relativePath: file.webkitRelativePath }));
  folderUploadBusy = true;
  el('uploadFolderBtn').disabled = true;
  try { await uploadFolderImages(images); }
  finally {
    folderUploadBusy = false;
    el('uploadFolderBtn').disabled = false;
    event.target.value = '';
  }
});
