/* 本地视觉训练平台 - 前端逻辑（纯 JavaScript，无需编译）
 *
 * 直接改本文件 → 刷新浏览器生效。
 * 注意：改完请把 index.html 里引用的 ?v=N 加 1（清浏览器缓存）。
 */

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
  node.textContent = '⚠ ' + msg;
}

window.addEventListener('error', (ev) => {
  banner(`脚本异常：${ev.message}（${ev.filename}:${ev.lineno}）`);
});

window.addEventListener('unhandledrejection', (ev) => {
  const r = ev.reason;
  banner(`异步异常：${r instanceof Error ? r.message : String(r)}`);
});

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

/* ==================== 标签页切换 ==================== */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');

    const name = tab.dataset.tab || '';
    el(`tab-${name}`).classList.add('active');

    if (name === 'detect') loadImages();
    if (name === 'video') loadVideos();
    if (name === 'train') loadDatasets();
  });
});

/* ==================== 环境状态 ==================== */

/* ==================== 环境状态 / 模型模式（普通 YOLO ↔ YOLO-World）==================== */

let WORLD_MODELS = [];   // 属于 YOLO-World 的权重文件名（由后端告知）
let CLIP_OK = false;     // World 所需的文本编码器依赖是否就绪

/** 填充下拉框：World 权重前加 🌍，并尽量保留原选择 */
function fillSelect(id, names) {
  const sel = el(id);
  const cur = sel.value;
  sel.innerHTML = '';
  if (!names.length) {
    sel.innerHTML = '<option value="">（无可用权重）</option>';
    return;
  }
  names.forEach((m) => {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = WORLD_MODELS.includes(m) ? `🌍 ${m}` : m;
    sel.appendChild(o);
  });
  if (cur && names.includes(cur)) sel.value = cur;
}

/** 切换"类别输入框"的显示，并更新模式提示条 */
function syncModeUi(selId, wrapId, hintId) {
  const sel = el(selId);
  const wrap = el(wrapId);
  const hint = el(hintId);
  const name = sel.value || '';
  const isWorld = WORLD_MODELS.includes(name);

  wrap.hidden = !isWorld;

  if (isWorld) {
    hint.className = 'mode-hint world';
    hint.innerHTML = CLIP_OK
      ? `🌍 <b>YOLO-World 模式</b>（${esc(name)}）：prompt 可用中英文随意写（如 <code>person, 书本</code>），留空则用权重内置类别`
      : `⚠ 已选 World 权重（${esc(name)}），但缺少 <code>clip</code> 依赖，无法使用。` +
        `请执行 <code>pip install "git+https://github.com/ultralytics/CLIP.git"</code>`;
  } else {
    hint.className = 'mode-hint';
    hint.innerHTML = `🔒 <b>普通 YOLO 模式</b>（${esc(name)}）：类别固定为权重内置的 COCO 80 类，不能用文字指定`;
  }
}

async function loadEnv() {
  const badge = el('envStatus');
  try {
    const s = await api('/api/yolo/status');
    badge.textContent = s.message;
    badge.className = 'badge ' + (s.available ? 'ok' : 'err');

    WORLD_MODELS = s.world_models || [];
    CLIP_OK = !!s.clip_available;

    const all = s.models.length ? s.models : ['yolov8n.pt'];
    fillSelect('detWeights', all);   // 图片检测：全部模型
    fillSelect('vidWeights', all);   // 视频检测：全部模型
    // 训练：排除 World 权重（训练走普通 YOLO 流程，喂 World 权重会失败）
    fillSelect('trainWeights', all.filter((m) => !WORLD_MODELS.includes(m)));

    syncModeUi('detWeights', 'detClassesWrap', 'detMode');
    syncModeUi('vidWeights', 'vidClassesWrap', 'vidMode');
  } catch (e) {
    badge.textContent = '状态获取失败：' + (e instanceof Error ? e.message : String(e));
    badge.className = 'badge err';
  }
}

// 切换模型 → 立即更新模式提示与类别输入框
el('detWeights').addEventListener('change', () =>
  syncModeUi('detWeights', 'detClassesWrap', 'detMode')
);
el('vidWeights').addEventListener('change', () =>
  syncModeUi('vidWeights', 'vidClassesWrap', 'vidMode')
);

/** 读取当前检测设置（图片/视频共用） */
function readDetectOptions(prefix) {
  const weights = el(`${prefix}Weights`).value || 'yolov8n.pt';
  const isWorld = WORLD_MODELS.includes(weights);
  const classes = isWorld ? el(`${prefix}Classes`).value.trim() : '';
  return { weights, isWorld, classes };
}

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
          await api(uploadPath, { method: 'POST', body: fd }, 120000);
          ok.push(f.name);
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

/* ==================== 图片检测 ==================== */

async function loadImages() {
  const box = el('imageGrid');
  try {
    const data = await api('/api/images');
    if (!data.images.length) {
      box.innerHTML = '<div class="empty">暂无图片，请先上传</div>';
      return;
    }
    box.innerHTML = '';
    data.images.forEach((img) => {
      const fig = document.createElement('figure');
      fig.title = '点击检测：' + img.filename;
      fig.innerHTML = `<img src="${esc(img.url)}" alt=""><figcaption>${esc(img.filename)}</figcaption>`;
      fig.addEventListener('click', () => detectImage(img.filename));
      box.appendChild(fig);
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

async function detectImage(filename) {
  const box = el('detectResult');
  let ticker = null;
  const t0 = Date.now();

  try {
    // 1) 先取参数（元素缺失会被 catch 住并显示出来）
    const conf = el('detConf').value || '0.25';
    const { weights, isWorld, classes } = readDetectOptions('det');
    const modeText = isWorld
      ? `🌍 World${classes ? `｜类别「${classes}」` : '｜类别=权重内置'}`
      : '普通 YOLO';

    // 2) 实时计时，区分"在跑"和"卡死"
    const paint = () => {
      box.innerHTML =
        `<div>检测中… <b>${((Date.now() - t0) / 1000).toFixed(1)}s</b></div>` +
        `<div style="font-size:12px;color:#80868b;margin-top:6px">` +
        `图片：${esc(filename)}｜置信度 ${esc(conf)}｜${esc(modeText)}<br>` +
        `（第一次检测要加载模型；World 切换类别还要编码文字；超过 300 秒自动报超时）</div>`;
    };
    paint();
    ticker = setInterval(paint, 200);

    // 3) 请求后端（图片检测允许长耗时）
    const url =
      `/api/detect/image/${encodeURIComponent(filename)}` +
      `?conf=${encodeURIComponent(conf)}&weights=${encodeURIComponent(weights)}` +
      (classes ? `&classes=${encodeURIComponent(classes)}` : '');
    const r = await api(url, { method: 'POST' }, 300000);

    // 4) 渲染结果（字段缺失时明确报错，而不是静默空白）
    if (!r || !r.result_url) {
      throw new Error(`后端返回数据不完整：${JSON.stringify(r).slice(0, 200)}`);
    }

    const summary = Object.entries(r.summary || {})
      .map(([k, v]) => `${k}×${v}`)
      .join('、');

    let html = `<img src="${esc(r.result_url)}?t=${Date.now()}" alt="检测结果">`;
    const clsList = ((r.classes_used) || []).map(String);
    const clsShort =
      clsList.length > 12
        ? clsList.slice(0, 12).join('、') + `…（共 ${clsList.length} 类）`
        : clsList.join('、');
    const modeLabel = r.mode === 'world' ? '🌍 World' : '普通 YOLO';
    html +=
      `<div class="summary">检出 ${esc(r.count)} 个目标${summary ? '：' + esc(summary) : ''}` +
      `<span style="color:#80868b;font-weight:400">（${esc(modeLabel)}｜当前类别：${esc(clsShort) || '?'}｜` +
      `耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s）</span></div>`;

    if (Array.isArray(r.detections) && r.detections.length) {
      html += '<table><tr><th>类别</th><th>置信度</th><th>位置 (x1,y1,x2,y2)</th></tr>';
      r.detections.slice(0, 100).forEach((d) => {
        html +=
          `<tr><td>${esc(d.class_name)}</td><td>${esc(d.confidence)}</td>` +
          `<td>${esc((d.bbox || []).join(', '))}</td></tr>`;
      });
      html += '</table>';
    }
    box.innerHTML = html;

    // 刷新左侧缩略图（失败不影响已显示的结果）
    try {
      await loadImages();
    } catch (e) {
      console.warn('[loadImages] 刷新失败', e);
    }
  } catch (err) {
    showError(box, err, '检测失败');
  } finally {
    // 无论成功失败都停掉计时器，避免"一直检测中"
    if (ticker !== null) clearInterval(ticker);
  }
}

el('refreshImages').addEventListener('click', loadImages);

/* ==================== 视频检测 ==================== */

async function loadVideos() {
  const box = el('videoList');
  try {
    const data = await api('/api/videos');
    if (!data.videos.length) {
      box.innerHTML = '<div class="empty">暂无视频，请先上传</div>';
      return;
    }
    box.innerHTML = '';
    data.videos.forEach((v) => {
      const item = document.createElement('div');
      item.className = 'item';
      item.innerHTML = `<span class="name">${esc(v.filename)}</span>`;

      const btn = document.createElement('button');
      btn.className = 'btn small';
      btn.textContent = '检测';
      btn.addEventListener('click', () => detectVideo(v.filename, btn));

      item.appendChild(btn);
      box.appendChild(item);
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败：${esc(e instanceof Error ? e.message : String(e))}</div>`;
  }
}

async function detectVideo(filename, btn) {
  const box = el('videoResult');
  let ticker = null;
  const t0 = Date.now();

  try {
    const conf = el('vidConf').value || '0.25';
    const stride = el('vidStride').value || '1';
    const { weights, isWorld, classes } = readDetectOptions('vid');
    const modeText = isWorld
      ? `🌍 World${classes ? `｜类别「${classes}」` : '｜类别=权重内置'}`
      : '普通 YOLO';

    btn.disabled = true;
    box.innerHTML =
      `<div>提交中… <b>${((Date.now() - t0) / 1000).toFixed(1)}s</b></div>` +
      `<div style="font-size:12px;color:#80868b">视频：${esc(filename)}｜置信度 ${esc(conf)}｜抽帧间隔 ${esc(stride)}｜${esc(modeText)}</div>`;

    // 提交任务：30 秒内必须返回 job_id
    const res = await api(
      `/api/detect/video/${encodeURIComponent(filename)}` +
        `?conf=${encodeURIComponent(conf)}&stride=${encodeURIComponent(stride)}` +
        `&weights=${encodeURIComponent(weights)}` +
        (classes ? `&classes=${encodeURIComponent(classes)}` : ''),
      { method: 'POST' },
      30000
    );
    if (!res || !res.job_id) {
      throw new Error(`后端未返回 job_id：${JSON.stringify(res).slice(0, 200)}`);
    }

    // 轮询期间保持计时
    ticker = setInterval(() => {
      const node = box.querySelector('.elapsed');
      if (node) node.textContent = `${((Date.now() - t0) / 1000).toFixed(1)}s`;
    }, 200);

    pollJob(
      res.job_id,
      box,
      (job) => {
        btn.disabled = false;
        if (job.result_url) {
          box.innerHTML =
            `<video src="${esc(job.result_url)}" controls></video>` +
            `<div class="summary">${esc(job.message)}</div>`;
        } else {
          box.innerHTML =
            `<div style="color:#c5221f;font-weight:600">视频检测未成功：${esc(job.message)}</div>`;
        }
        loadVideos();
      },
      t0
    );
  } catch (err) {
    btn.disabled = false;
    showError(box, err, '视频检测失败');
  } finally {
    if (ticker !== null) clearInterval(ticker);
  }
}

el('refreshVideos').addEventListener('click', loadVideos);

/* ==================== 训练 ==================== */

async function loadDatasets() {
  const box = el('datasetList');
  const sel = el('trainDataset');
  try {
    const data = await api('/api/datasets');
    sel.innerHTML = '';
    if (!data.datasets.length) {
      box.innerHTML = '<div class="empty">暂无数据集</div>';
      sel.innerHTML = '<option value="">（无数据集）</option>';
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

      const o = document.createElement('option');
      o.value = d.name;
      o.textContent = d.name;
      sel.appendChild(o);
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
    weights: el('trainWeights').value || 'yolov8n.pt',
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

/* ==================== 通用：轮询后台任务 ====================
 * - 单次查询 15 秒超时
 * - 偶发失败自动重试（连续 5 次才判失败）
 * - 一直显示已用时，避免"看起来卡死"
 */
function pollJob(jobId, box, onFinish, t0 = Date.now()) {
  let failCount = 0;

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
        `｜已用时 <span class="elapsed">${elapsed}s</span></div>`;

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
          `<div style="font-size:12px;color:#80868b">已用时 <span class="elapsed">${elapsed}s</span></div>`;
      }
    }
  }, 1500);
}

/* ==================== 初始化 ==================== */
loadEnv();
loadImages();
