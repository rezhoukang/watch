# 本地视觉训练平台

单机自用（仅监听 `127.0.0.1`）的视觉工具：图片/视频上传、YOLOv8 推理、YOLOv8 训练。
FastAPI 后端 + 原生 HTML/CSS/JS 前端，数据存本地文件，**无需数据库**。

## YOLOv8 权重下载

下载后按下表文件名放入项目根目录的 `weight/` 文件夹，页面会扫描该目录生成权重选项。

| 权重文件 | 用途 | 官方下载地址 |
| --- | --- | --- |
| `medium.pt` | Head 人头检测，目标检测融合的基准模型 | [原作者仓库下载 medium.pt](https://raw.githubusercontent.com/Abcfsa/YOLOv8_head_detector/main/medium.pt) |
| `yolov8s.pt` | COCO 通用目标检测，Small | [下载 yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) |
| `yolov8m.pt` | COCO 通用目标检测，Medium | [下载 yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) |
| `yolov8m-pose.pt` | 人体与关键点检测 | [下载 yolov8m-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt) |
| `yolov8s-worldv2.pt` | 开放词表目标检测，Small | [下载 yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt) |
| `yolov8m-worldv2.pt` | 开放词表目标检测，Medium，Pose 物体关联使用 | [下载 yolov8m-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-worldv2.pt) |
| `yolov8l-worldv2.pt` | 开放词表目标检测，Large | [下载 yolov8l-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-worldv2.pt) |

模型说明和其他规格权重见 [Ultralytics YOLOv8 官方文档](https://docs.ultralytics.com/models/yolov8/)及 [YOLO-World 官方文档](https://docs.ultralytics.com/models/yolo-world/)。

### 目标检测融合使用的 medium.pt

本项目的人头融合检测读取 `weight/medium.pt`，来源为 [Abcfsa/YOLOv8_head_detector](https://github.com/Abcfsa/YOLOv8_head_detector)。原作者说明该模型使用 YOLOv8，在 SCUT-HEAD 的 Part A、Part B 数据集上训练。可从[原始文件页面](https://github.com/Abcfsa/YOLOv8_head_detector/blob/main/medium.pt)或[直接下载地址](https://raw.githubusercontent.com/Abcfsa/YOLOv8_head_detector/main/medium.pt)获取，并保存为 `weight/medium.pt`。

2026-11-07 核对结果为仓库文件与本地权重逐字节哈希一致，文件大小 `52,020,374` 字节，SHA256 为 `9fd59f93bdd43c3369915aa0ecf2843bb163292cbb41c431e3448a68f2ea4ce2`。

## 人脸识别

「图片推理」与「视频推理」之间新增「人脸识别」。左侧复用待推理图片列表，支持批量上传、单张删除、多选删除、多选、全选及批量推理；上传和删除会同步影响图片推理的待推理图库，已保存批次中的裁剪图继续保留。固定调用图片推理的目标检测融合（medium 0.45 + RT-DETRv4 0.25、输入640），可调合并方向、IoU和生效置信度。右侧「头管理」保存每个批次的所有有效检测框裁剪，不依赖人脸检测或清晰度筛选。

每个裁剪具有 `HC001-001` 形式的批次内编号，保存无标记原始 PNG、带编号底栏的 PNG、原图编号总览和坐标/类别/置信度。支持按来源图片筛选、搜索编号/类别、分页查看、取消尚未处理的图片；重启后仍可查看已生成结果。

人脸识别页面使用紧凑缩略图，待切割图片、原图编号总览和 HC 裁剪图支持悬停大图预览。图片、卡片文字和操作按钮居中；窄屏下两侧面板改为上下排列。

文件集中保存于 `output/head_cuts/<批次编号>/`。`.gitignore` 忽略 `/output/`、`/head/`、`/tmp/`，并继续忽略 `/data/`，生成的头像、课堂图片、批次文件和 ZIP 均不进入 Git。接口：`POST /api/head-cuts`（filenames、rtdetr_base、merge_iou、merge_confidence），`GET /api/head-cuts` 列表，`GET /api/head-cuts/<批次编号>` 详情。

在「头管理」选择已完成的批次，点击「批量检测人脸」，使用 SCRFD-10G 对该批次 `crops/` 中的原始 PNG 检测，输入640、默认置信度0.5，界面可调0.01～0.99。保留每个 `HC…` 父编号；同一裁剪多张人脸使用 `_F01`、`_F02` 后缀。结果显示检测状态、置信度、人脸原始裁剪及检测框，支持检出/未检出/失败/未检测筛选、取消任务及重启后恢复查看。

人脸结果保存在同一批次的 `face_detection.json`、`faces/<任务编号>/` 和 `face_annotations/<任务编号>/`，自动包含在批次ZIP中，仍由 `/output/` 规则排除Git。JSON保存裁剪内和原课堂图片的人脸框坐标、五点位置及置信度。此步骤是人脸检测；原始Head裁剪继续保留。可选依赖：`pip install -r requirements-face-detection.txt`，检测权重放在 `weight/scrfd/det_10g.onnx`。接口：`POST /api/head-cuts/<批次编号>/faces`，请求体 `{"confidence":0.5}`。

### 1:N 人脸识别与花名册

管线的最后一段是身份识别：YOLOv8 + RT-DETRv4 融合出 Head ROI，SCRFD 在 ROI 内检出并裁剪人脸，`align_face` 按五点对齐，再用 ArcFace 提取 512 维特征，最后与学生人脸库做 1:N 匹配。识别模型是 insightface buffalo_l 的 `w600k_r50.onnx`（与已有的 `det_10g.onnx` 同包），放在 `weight/scrfd/w600k_r50.onnx`，用 onnxruntime CPU 推理，不新增依赖。

「头管理」提供「1:N 识别」按钮，对当前批次已检出的人脸批量识别。三个门槛都可调，**不写死在代码里**：识别阈值（默认 `0.4`）、质量置信度（默认 `0.5`）、最小人脸（默认 `24` px）。结果写入批次的 `recognition.json`，包含每个人的命中姓名、相似度、候选列表、被拒原因，以及**人脸在原始像素下的实际尺寸** `face_pixels`。人脸卡上按状态显示徽标：命中为绿色「姓名 + 相似度」，质量不足为黄色「质量不足」（悬停显示具体原因），其余为「未识别」（悬停显示原因与最高分）。「全部识别结果 / 已识别 / 未识别 / 未识别任务」可筛选。

**识别是一对一的**：同一个库内编号只会给一张脸，不会出现“两张脸都是同一个人”。实现上把所有 (脸，候选) 组合按相似度降序处理，**相似度高的先占**；没抢到的脸自动顺延到它的下一个候选，再被抢就继续顺延（第 2、第 3……直到没有可用候选）；若所有候选都低于识别阈值或都已被占走，该脸判为未识别。只有相似度 ≥ 识别阈值的候选才参与分配——低于阈值的候选本来就不可信，不会被拿来补位。

界面上顺延到备选的脸会标注「备选#2」/「备选#3」，未识别的脸在提示里写明原因（是“候选被得分更高的另一张脸占用”还是“候选相似度不足”），并说明被哪张脸抢走。记录里的 `reassigned` 是顺延到备选的数量，`conflicts` 是被抢占过的脸数，都写在完成消息里。每个脸保留前 5 个候选（`CANDIDATE_K`）供消解使用，同时便于人工复核。

**命中的人脸会在右侧一并显示人像库里对应的登记照**（`→` 指向库内照片），方便直接人工核对是不是同一个人；登记照与库内编号、（可选）姓名都写在提示里。未命中项不显示，避免造成误读。

人脸库是动态可增删的：`GET /api/face-library` 列表，`POST /api/face-library`（`name` + 图片）添加一个人，`DELETE /api/face-library/<编号>` 删除，`GET /api/face-library/<编号>/photo` 取该人的登记照，`GET /api/face-library/status` 查看模型与库状态（含门槛默认值），`POST /api/face-library/import-students` 一次性导入 `output/pdf/2026_software_students/` 的 155 张学生登记照（姓名优先取文件名 `001_示例姓名.jpeg`，回退 `names.json`）。数据存于 `output/face_library/`：`library.json`（编号/姓名/来源）、`embeddings.npz`（512 维特征）、`photos/<编号>.jpg`（**对齐后的人脸图**，即识别结果里右侧显示的登记照），同样由 `/output/` 规则排除Git。删除某人时其登记照一并删除。导入按编号幂等：已入库且有登记照的跳过；**已入库但缺照片的旧条目会自动回填照片**，不重复写入特征。识别接口：`POST /api/head-cuts/<批次编号>/recognize`，请求体 `{"threshold":0.4,"min_confidence":0.5,"min_size":24,"min_sharpness":50}`。

三点必须注意，否则容易高估结果：

1. **质量置信度默认与检出阈值同源（0.5），不是 0.8**。`assess_quality` 的置信度门限原本硬编码 0.8（匿名人脸复核的基准，该流水线仍保持此默认值）；但识别的检测本身按 0.5 跑，再要求 0.8 等于重复设卡——实测 A/B 教室 172 张脸的 SCRFD 置信度中位 0.582、**最高仅 0.774**，卡 0.8 会让**全部人脸进不了比对**，命中恒为 0。因此识别端默认取 0.5，并可通过 `min_confidence` 调高。
2. **24 像素只是下限，不代表可识别**。SCRFD 把输入缩放到 640×640、ArcFace 用 112×112，都不意味着原图里 24 像素的人脸能被可靠辨认。A/B 教室实测人脸短边中位数仅 24~26 像素（最大约 50），跨域到高清登记照后相似度普遍只有 0.15~0.35，而不同登记照之间的区分度也在这个量级。实测把 `min_size` 从 24 放宽到 16/12，进入比对的人脸从 71 增到 133/137，但命中数始终是 2，且相似度中位数反而从 0.243 降到 0.221/0.214——多出来的只是噪声。结果里的 `face_pixels` 就是用来判断某个命中是否可信的。
3. **建库与查询的门槛不同**。花名册导入是权威基准，只要求 SCRFD 检出人脸（置信度 ≥0.5）即可入库，不套用查询端的质量门槛。

识别的坐标基准是 **HC 裁剪图**：`face_detection.json` 里的人脸框与五点都是相对该 Head 裁剪图的坐标（与原始课堂图片 1:1 同尺度），所以识别必须打开 `crops/<HC编号>.png`，不能用单独裁出的人脸小图 `faces/…png`——后者坐标系不一致，会导致清晰度取样区域和对齐都错位。



## 独立动作识别（Head → 区域裁剪 → VLM）

顶部「动作识别」是独立模块，保留原有图片、视频、Pose、数据集和聚合推理入口。新流程直接复用 Head 目标检测，以每个头框的宽高向左右、顶部和下方扩展，从原始图片矩形裁剪人物及附近桌面；不调用 Pose，也不接入姓名或人像库。保存干净裁剪和仅标记目标头框的 VLM 输入，保留矩形内的场景，不加外侧遮罩。

1. 选择或上传课堂图片，点击「Head 检测并生成裁剪」。Head 使用现有 medium 0.45 + RT-DETRv4 0.25、输入 640、合并 IoU 0.1、生效置信度 >0.55。
2. 调整左右扩展、顶部留白、向下延伸，默认及推荐比例分别为每侧 `0.85` 倍头宽、`0.00` 倍头高、头框下沿以下 `1.30` 倍头高。滑块预览当前目标，点「重新生成本批次裁剪」保存全部目标；也可用原图拖框或原图像素坐标单独修正，区域必须包含目标头框。区域只是几何估算，遮挡和邻座干扰需要检查。
3. 点击「分析全部裁剪」，或选中部分目标后分析。每人只输出抬头、低头、低头玩手机、低头看书、睡觉之一，证据不足时输出无法判断，同时保存可见证据和不确定原因。睡觉按明显趴桌或闭眼伏下等可见姿态判断，不能仅凭低头或遮挡推断已经入睡；出现物品本身不等于使用该物品。取消会在当前目标结束后停止，已完成结果保留。

默认本地模型为 **Qwen3-VL-4B-Instruct / 4-bit NF4**，可切换 2B 或 FP16。首次安装额外依赖：`.venv\Scripts\python.exe -m pip install -r requirements-vlm.txt`。页面可检查、下载模型和释放 VLM 显存，权重保存到 `weight/Qwen3-VL-4B-Instruct/` 或 `weight/Qwen3-VL-2B-Instruct/`；4-bit 为运行时量化，官方下载的原始 4B 权重磁盘占用约 8.9 GB。只从本地权重加载推理，下载模型期间不上传课堂图片。串行处理裁剪，VLM 输入保持比例、最长边不超过 768 像素。进入 Head 检测前卸载 VLM，进入 VLM 分析前释放 Head 模型缓存，以减少显存争用。

批次原图快照、干净裁剪、标记目标头框的 VLM 输入和 `record.json` 保存于 `data/outputs/action_recognition/<批次编号>/`。`data/`、`weight/` 已在 `.gitignore` 中，图片和权重不会提交 Git。区域修改会清除对应旧动作；相同裁剪、模型、量化设置和提示词版本可复用已有结果，勾选「重新分析已有结果」可强制重跑。历史批次在页面左侧恢复。

页面的「动作与提示词配置」支持添加动作、启用/停用、编辑判定规则及 0～100 的优先级，也提供「启用打哈欠示例」、JSON 编辑和完整提示词预览。默认仍启用原有五类动作，打哈欠示例初始关闭；“无法判断”始终保留。配置按 JSON 保存，后端组装为文本任务指令发送给 VLM，输出也要求 JSON（`state`、`evidence`、`uncertainty`），不需要重新训练模型；新增动作的实际效果需要课堂样本验证。规则只判断可见动作，不依据打哈欠推断疲劳。

可随代码查看和修改的默认配置是 `action_recognition/default_prompt.json`，前端保存的配置位于已忽略的 `data/outputs/action_recognition/prompt_config.json`。`GET/PUT /api/actions/prompt` 读取/保存，`POST /api/actions/prompt/preview` 只编译预览、不保存。每次分析启动时固定一份配置，保存期间的修改只影响下一次任务；`record.json` 中的 `prompt_profiles` 保存配置与完整提示词，动作结果通过 `prompt_config_id` 对应。规则改变会使相同裁剪的旧缓存失效，但不会直接修改历史动作结果。

后端拆分为 `action_recognition/regions.py`（几何区域）、`store.py`（持久化和裁剪）、`prompts.py`（JSON 配置及提示词组装）、`vlm.py`（本地模型和结果校验）、`jobs.py`（批量后台任务），独立接口在 `web/actions.py`，页面脚本为 `frontend/js/actions.js`、`frontend/js/action-prompts.js`，样式为 `frontend/styles/actions.css`。接口统一前缀 `/api/actions`，独立于原有推理接口。

## 聚合推理

「数据集管理」右侧新增「聚合推理」标签页，布局与「图片推理」一致：顶部工具栏 → 子标签（待推理图片 / 聚合结果）→ 图库或结果区。左侧图库与图片推理共用同一个 `data/uploads/`，点一张图即按顺序跑三步，结果区在原图上叠框显示。

1. **目标检测**：与「图片推理 → 目标检测」完全同一套——`medium.pt` 0.45 为基准、RT-DETRv4 0.25 补齐，两路 imgsz 固定 640，合并 IoU ≤ 0.1、自定义 IoU 生效置信度 > 0.55。这些参数在 `aggregate.py` 里与 `head_fusion` 默认值同源锁死，**接口与界面都不暴露**，避免两处目标检测参数各自漂移。
2. **人脸识别**：整图 SCRFD 检出所有人脸 → 质量筛选 → ArcFace 提特征 → 与学生人脸库 1:N 比对，然后**把识别到的姓名贴到对应的人头框上**（人脸框落在人头框内时 IoU = 人脸面积 / 人头面积 > 0）。一个库内编号只给一张脸（复用一对一贪心消解），一个人头也只贴一个姓名。

**人脸识别参数全部锁死**（`aggregate.FACE_CONFIDENCE` / `MATCH_THRESHOLD` / `MIN_QUALITY_CONFIDENCE` / `MIN_FACE_PIXELS` / `MIN_FACE_SHARPNESS`）：人脸置信度 `0.5`、识别阈值 `0.3`、质量门槛 `0.5 / 24px / 50`。界面上不再有阈值输入框，只剩「人脸识别」「姿势检测」两个勾选框（取消勾选即跳过该步，`steps.<步骤>.enabled=false`）；接口也只收 `face` / `pose`，多传的阈值字段一律忽略。生效值回报在 `steps.face` 里，方便核对。远景小脸给的阈值太低会放进模糊/侧脸，太高又会把整排人都卡掉，所以把这两个值钉死在这里。
3. **姿势检测**：固定使用 `yolov8m-pose.pt`，人体置信度 `0.6`、imgsz `640`、低头 `R < 0.05`、抬头 `R > 0.20`。复用图片 Pose 推理流程，举手优先，排除无法判断者；开启全图物体检测与关联（Pose 框），物体置信度固定 `0.03`。过滤后的姿势再与 head 框绑定；这些参数不开放调整，旧客户端提交的姿势参数也会被忽略。

**绑定规则（固定，不对外暴露）**：只保留 IoU > 0 的配对，所有配对按 IoU 降序逐个占位（**IoU 相同时中心距离近的先占**），一个人头最多绑一个姿势、一个姿势最多绑一个人头（链式贪心 + 一一对应）。人头框远小于人体框，但人头在人体框内 → IoU = 人头面积 / 人体框面积 > 0，所以「IoU > 0」既筛掉了完全不沾边的组合，又不用再调阈值。行为沿用 Pose 默认阈值（低头 R < 0.05、抬头 R > 0.20）。

「人脸识别」与「姿势检测」各有一个勾选框，取消即跳过该步：记录里 `steps.<步骤>.enabled=false`，界面显示「未启用」，不会让人误以为跑过。

聚合结果提供「常显标签」开关：勾选后所有目标标签始终显示，取消后只在悬停时显示；切换结果图片时保留当前选择。

**渲染与图片推理一致**：结果区是原图 + 前端叠框（框用百分比定位，缩放自适应），**只画 head 框**，姓名、相似度与姿势状态都挂在这个框的标签上；框下方是与图片推理同款的胶囊筛选（全部 head / 已贴姓名 / 未识别 / 已绑姿势）与明细表（人头、姓名、姿势 + 绑定 IoU、中心、左上右下）。未归属的人脸与姿势会单独列出，不会硬塞给某个人头。后端另存一份对照图（同样只画 head 框，标签烧进像素）在 `data/outputs/aggregate/<原名>_agg.<后缀>`；中文姓名用 Windows 自带字体（`msyh.ttc` / `simhei.ttf` 等）绘制，找不到字体时退化成库内编号 `#0001`，不会画出方块。

接口：`POST /api/aggregate/image/<文件名>`，请求体只含 `face` / `pose` 两个开关（目标检测、人脸阈值、姿势配置与绑定规则全部锁死在后端，多余字段被忽略）；后端对照图经 `GET /api/aggregate/results/<文件名>` 取回。返回里 `targets` 是人头（含 `name`、`entry_id`、`similarity`、`poses`），`faces` 是人脸（含 `bbox`、状态、候选、被抢占的候选），`poses` 是人（含骨架与绑定依据），另有 `steps`（每步统计、耗时与目标检测 `rule`，姿势固定参数和 `excluded_unknown`）、`bind`（绑定规则说明）、`summary`（汇总）、`unbound_faces` / `unbound_poses`（未归属清单）。

## 一键 head 检测融合

图片推理、视频推理和标注器预标注的权重右侧均有“head检测”开关。开启后固定使用 `medium.pt`、置信度 `0.45`
作为基准，以及 `best_stg2.safetensors`（RT-DETRv4）、置信度 `0.25` 补充人头，
两路输入尺寸均为 `640`。关闭开关恢复原来的权重、置信度和尺寸选择。
head 检测默认合并 IoU 为 `0.1`，生效置信度门槛为 `0.55`：待补入框置信度 > 0.55 时允许 IoU ≤ 0.1；置信度 ≤ 0.55 时仍要求 IoU=0。图片、视频、预标注以及正反向 head 融合均使用该默认值；手动历史记录合并仍按其单独设置。
开启 head 检测后可勾选「以 RT-DETR 为基准」：保留 RT-DETRv4 0.25 的全部结果，用 medium 0.45 补齐。图片、视频和预标注共用此选项，记录中保存合并方向，绿框始终是基准、橙框是补充。接口参数为 `head_detection=true&rtdetr_base=true`。

保留全部基准框。「合并 IoU ≤」可设为 0～1（默认 0，步长 0.01），候选框与任一原始基准框的 IoU **超过**阈值时舍弃，否则补入，包括等于阈值的情况。阈值越大，允许的重叠越多；例如 0.2 允许 IoU ≤ 0.2 的候选框补入。
跨类别同样适用，补充框之间不额外去重。默认 0 保持原有规则。图片、视频和预标注接口使用 `merge_iou=0.2`；历史记录保存实际阈值，视频文件名包含非零阈值以便对比。
「自定义 IoU 生效置信度 >」可设为 0～1（默认 0）。只检查待补入框的置信度：**高于**门槛才使用自定义 IoU，低于或等于门槛仍要求与所有基准框 IoU=0；低置信度框没有重叠时依然可补入。此规则不删除基准框，正反向融合、视频、预标注和手动合并均适用。接口参数为 `merge_confidence`，保存到记录及视频参数；例如 `merge_iou=0.2&merge_confidence=0.5`。
融合保留原图像素坐标 `(x1, y1, x2, y2)` 和类别名，统一数值类别 ID，并保留 `source_class_id`、`source_model`。
基准框绿色、补充框橙色，结果页初始显示全部融合框。涉及非人头类别时，类别统计分别保留。
每次融合自动保存为“合并图”记录，JSON 包含两路参数、补充/舍弃数量和来源信息。

接口：`POST /api/detect/image/<filename>?head_detection=true`。开启时固定两路参数，
传入的 `weights`、`conf`、`classes`、`imgsz` 不会改变这套融合配置。两路模型均须可用。

视频接口 `POST /api/detect/video/<filename>?head_detection=true` 按原有抽帧间隔和时间段运行，
每个抽样帧的两路结果都使用与历史记录合并相同的 `merge_detections` 函数融合。
输出文件名含 `_head`，绿框为 medium 基准、橙框为 RT-DETRv4 补充；框/中心点模式均可用。
人数曲线统计融合后的 head 类数量。视频结果库保留融合模式和两路参数；“图预览”会同步图片推理开关。
融合视频仍在后台保存 JSONL（界面不提供逐帧坐标按钮）：第一行是参数、fps、stride 与起始帧，其余行是每个实际推理帧的
绝对帧号、时间、`detections`（坐标/类别/置信度/模型来源）及融合统计；未推理帧沿用上一推理帧的渲染图。
文件保存在 `data/outputs/detections/`，通过 `/outputs/detections/<filename>` 下载，删除输出视频时一起删除。

标注预标注接口 `POST /api/annotate/prelabel/<dataset>/<split>/<image>?head_detection=true`
先融合，再按类别名映射到数据集 class_id；仍可手动修改并保存 YOLO 标签。

## 推理结果图片记录与合并

### Pose 骨架显示

关闭 head 检测，选择 `yolov8m-pose.pt` 后运行图片或视频推理，可显示人体框及上半身 11 点骨架（鼻、眼、耳、肩、肘、腕）。图片结果提供「上半身骨架」显示开关，随类别、人体框置信度和去重筛选同步更新。青色为连接线，粉色为关键点，仅绘制关键点置信度 ≥ 0.5 且在图像范围内的点及有效连线；髋、膝、踝不显示。
YOLOv8m Pose 默认人体检测置信度为 `0.6`，物体检测置信度为 `0.03`，图片、视频和预标注入口一致。两项均可手动调整；切换回普通检测权重时恢复原来的置信度设置。
每个人的 `detections[].keypoints` 保存为 17 组原图像素 `[x, y, confidence]`，`pose` 字段保存关节点名称和连接关系；历史结果 PNG 与视频输出均包含骨架。Pose 视频曲线按人体数量统计。当前提供上半身骨架与举手、低头、抬头规则；Head 匹配为后续功能。

### 行为标注与举手规则

Pose 低头状态继续使用全图物体检测与最大交集关联：直接以 Pose 人体目标框 `(x1,y1,x2,y2)` 作为人员 ROI，不再使用六点凸包，关联不依赖肘、腕关键点是否可见。整张原图只送入独立的 `yolov8m-worldv2.pt` 一次，固定提示词为 `cell phone`、`book`，物体置信度默认 ≥0.03。每个检出的物体与所有可关联的低头人员 ROI 计算交集面积，只绑定给交集面积最大且 >0 的那个人；交集并列时按人员序号确定唯一归属。采用交集面积而非 IoU，不再要求物体中心位于 ROI 或 60% 框面积位于 ROI。重复书本框先按置信度去重。

绑定手机优先判为“疑似看手机”，否则绑定书本判为“疑似阅读”；没有可靠物体、没有正交集、Pose 框无效或检测不可用时保留“低头”。举手、抬头不参与物体关联，每人仍仅有一个状态。识别物体只能支持疑似行为，不能确认视线或用途。

图片、视频和预标注的 Pose 设置提供「全图物体检测与关联」开关和物体置信度。结果中的 `behavior.roi` 保存 Pose 框矩形、原图物体坐标、置信度、交集面积和绑定人员序号；前端骨架层显示橙色 ROI 边界与黄色物体框，结果图及视频也绘制这些证据。模型未加载成功时图片结果会提示，详细原因保存在关联状态中。人工行为标注类别暂保持原有列表。

Pose 推理使用 `pose_behaviors.py` 的举手规则：双肩及同侧肘腕关键点置信度 ≥0.6，手腕高于同侧肩至少 0.25 倍肩宽、高于肘至少 0.15 倍肩宽，并高于可靠头部位置至少 0.15 倍肩宽。头部参考依次使用鼻、双眼、双耳。任一侧满足即为“举手”，每人仅有一个状态。

未判为举手时，鼻子和双肩关键点均须置信度 ≥0.6、坐标有效，且肩宽 ≥1 像素。计算 `R=(双肩中心Y-鼻子Y)/双肩欧氏距离`。低头和抬头分别使用独立阈值，默认 R<0.05 判低头，R>0.20 判抬头；中间区域（含两个边界）与关键点不足者均排除。举手始终优先。选择 Pose 权重后，图片、视频及预标注工具栏提供两组滑块和数值输入（-1～1，步长 0.01），低头阈值必须小于抬头阈值，重新推理生效。被排除者不进入框、骨架、结果坐标、人数和视频曲线；`excluded_unknown` 记录图片排除数。

这是用三张东教室图片初步校准的几何估计（原单阈值方案：31个人工样本，29人可计算，其中28人与标注一致；双阈值方案尚需独立样本验证），存在戴帽、遮挡等误判，需要更多图片验证。阈值、比例及判断依据保存在 `detections[].behavior`，生效阈值也保存在 `pose.head_rule` 和记录的 `parameters.head_down_r_threshold` / `parameters.head_up_r_threshold`；前端显示状态及分类人数，图片和视频输出显示英文状态。旧历史结果和人工标注不会自动改写。标注数据集仍让非举手样本保持“待标注”，避免自动规则污染人工校准数据。

数据集管理中点「导入东教室 Pose 结果」，或在推理历史选中 Pose 记录点「送去行为标注」。导入仅保留人体置信度 >0.7 的框，自动举手为“举手”，其他为“待标注”。打开原有全屏标注器，可看到上半身骨架，在框的下拉框选择低头、抬头、正常、无法判断或举手；每人只有一个状态，允许人工纠正规则误判。选中框后，快捷键 1～5 对应低头/抬头/正常/无法判断/举手，修改后点保存。

行为数据集使用固定状态类别。YOLO TXT 保存人体框和单一状态，`behavior/train/*.json` 保存稳定人员编号、17 点原始坐标及置信度、原始人体框、人工标签、标签来源和源推理记录。重复导入同一记录会跳过，保留人工修改。点「导出行为标注 JSON」即可获得后续调规则的数据，未标注样本仍明确保留“待标注”。普通数据集的原有标注流程保持不变。

「图片推理」现在有「待推理图片 / 推理结果 / 推理结果图片」三个子页面。
现有结果查看器继续支持框、实心、中心点、类别筛选和置信度筛选。

- 每次网页图片推理都生成独立记录，存于 `data/inferences/<记录ID>/`，包含原图快照、
  带框 PNG、时间、模型、置信度、推理尺寸、类别、坐标、统计及耗时。
  同名图片再次上传或删除待推理图片，不影响已保存记录。
- 点击「推理结果图片」中的缩略图，重新打开该次结果，无需重跑模型；可下载完整 JSON。
- 下方图片集可按“推理结果图 / 合并图”、模型权重和图片名称筛选；全选与批量删除只作用于当前筛选结果。
  记录从本功能启用后开始保存，旧版未归档的结果不会自动补成历史记录。
- 合并区左右分别选择「基准记录」和「待合并记录」，可点「交换方向」。只允许同一原图
  （像素内容、尺寸一致）的不同记录；待合并下拉框自动筛选兼容记录。
- 所有基准框保留。手动合并同样可设置「合并 IoU ≤」，待合并框与任一基准框的 **IoU 超过阈值**时舍弃，
  不比较置信度，也不限定同类别；仅边缘相碰保留。待合并内部沿用其已有框。
  点击「合并并保存新记录」后生成新的记录，两个来源记录均保留。
  合并结果中基准框为绿色，新增框为橙色，JSON 保存两个父记录和合并统计；合并结果可继续参与合并。
- 记录选择框按「文件名 / 模型 / 数量 / 置信度 / 时间 / 编号 / 颜色」显示；合并记录列出来源置信度。
  待推理缩略图、历史结果缩略图及合并预览支持悬停放大；“推理结果”详情大图不放大。
- 待推理图库和历史图库支持勾选、全选、取消选择与批量删除，也可点缩略图右上角 `×` 单独删除。
  当前结果详情也有删除入口。删除待推理图片不影响历史快照；删除一条历史记录会删除
  该记录的快照、带框图及 JSON，不会连带删除原始上传图片、其他记录或已经另存的合并结果。

接口：`GET /api/inferences` 列表，`GET /api/inferences/<ID>` 详情，
`POST /api/inferences/merge` 的 JSON 为 `{"base_id":"...","incoming_id":"..."}`。
删除记录使用 `DELETE /api/inferences/<ID>`；批量删除使用 `POST /api/inferences/delete`
（`{"record_ids":["..."]}`）或 `POST /api/images/delete`（`{"filenames":["..."]}`）。
批量接口返回 `deleted`、`missing`、`failed`，失败项保留，界面可再次选择重试。

## 课堂图片人数统计：RT-DETRv4-S C-RADIOv4

图片推理已支持 [Awiros 发布的 CrowdHuman 人体/人头模型](https://huggingface.co/Awiros/person_and_head_detection)，
使用 [RT-DETRv4 官方引擎](https://github.com/RT-DETRs/RT-DETRv4) 加载学生权重。
发布模型的类别是 `0=person`、`1=head`；本项目仅输出 `head`，以检出人头数估计人数，避免人体和人头重复计数。
推理无需下载 C-RADIOv4 教师模型，也无需把 safetensors 转成 YOLO 权重。

首次安装（在项目根目录，已有的 torch/torchvision 继续复用）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rtdetrv4.txt
.\.venv\Scripts\python.exe scripts/setup_rtdetrv4.py
```

安装脚本获取固定版本的官方源码到 `third_party/RT-DETRv4/`，以及约 42MB 的权重到
`weight/best_stg2.safetensors`。已有权重通过 SHA256 校验后直接复用；不覆盖不匹配的文件。
源码固定为 `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`，模型发布版本为
`823ca54bbf1446173f6906198a5e8b2a3e090890`。
两者均不提交到项目 Git；换机时运行上述命令即可恢复。
如已有同版本官方源码，可设置环境变量 `RTDETRV4_REPO` 指向该目录。
缺依赖或源码时 Web 服务仍可启动，选择该模型会显示安装提示。

启动或重启 `start.bat`，刷新浏览器，在「图片推理」选择
**RT-DETRv4-S C-RADIOv4（人头计数）**，点课堂图片即可查看人数、检测框和坐标明细。
安装就绪时图片页面默认选择此模型；视频权重下拉框也可选择该模型，输入尺寸固定 640，支持抽帧、时间段、框和中心点模式。
模型固定检测人头，无需填写 prompt。支持 CPU / CUDA 自动选择，并缓存模型，后续图片复用。

默认可从 `conf=0.38`、`imgsz=640` 开始。远处人头较小时可试 `960` 或 `1280`；
降低阈值提高召回，也会增加误检。结果页的置信度筛选和去重会改变「当前筛选」人数，
原始模型检出数会同时保留显示。应结合有人工人数标注的课堂样图调整阈值；检出人头数不代表已验证的真实出勤人数。
模型默认最多返回 300 个候选（人体和人头共用），超密集图片可能受此上限影响。

命令行也可统计（输出带框图片和 UTF-8 JSON 到 `data/results/`）：

```powershell
.\.venv\Scripts\python.exe scripts/count_heads.py "data/uploads/东教室阶梯.png" --conf 0.38 --imgsz 960
```

沿用原图片接口，指定权重即可：

```text
POST /api/detect/image/<图片文件名>?weights=best_stg2.safetensors&conf=0.38&imgsz=640
GET  /api/rtdetrv4/status
```

返回兼容原来的 `count`、`summary`、`detections`、`bbox`、`bbox_cxcywh` 和结果图 URL，
另外提供 `person_count` / `head_count`（均为人头数）、`mode="rtdetrv4"` 和 `device`。
坐标是原图像素，框会裁剪到图像边界。结果图用 `<原文件名>_rtdetrv4_head.jpg`，
与 YOLO 结果分别保存（此文件名用于命令行入口；网页结果独立归档到 `data/inferences/`）。
当前接入范围为图片人数统计；视频、训练和标注器接口会拒绝该权重。

回归检查（需已安装基础项目依赖及 torch/torchvision；TestClient 需要 httpx）：

```powershell
.\.venv\Scripts\python.exe -m pip install httpx
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node tests/test_history_ui.js
node --experimental-vm-modules tests/test_frontend_modules.cjs
```

## 环境（重要）

venv 用 `--system-site-packages` 创建，**torch / torchvision 直接复用系统 Python 里已装的版本**，
不重复安装（省 ~4GB）：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # 只装 ultralytics 等
```

当前环境实测：Python 3.12.10 · torch 2.11.0+cu128（来自系统）· ultralytics 8.4 · RTX 3060 可用。

## 使用

双击 `start.bat` → 浏览器打开 http://127.0.0.1:8000 ，共 7 个标签页：

| 标签 | 功能 |
|------|------|
| 上传 | 拖拽/选择图片、视频上传；**沿用原文件名**（同名直接覆盖，非法字符会自动替换成 `_`） |
| 图片推理 | 分「待推理图片 / 推理结果」两个子页面：点图 → 自动跳到结果页看大图 + 类别/置信度/坐标表；缩略图右上角的 `×` 可删除该图片 |
| 视频推理 | 选视频 → 后台逐帧推理 → 播放带框结果视频（可设抽帧间隔提速）；结果库每项有「图预览」（提取该段检出目标数最多的一帧到「待推理图片」并自动推理）|
| 训练 | 选数据集 → 建参数 → 后台训练 → 轮询进度 → 下载 best.pt；含**内置全屏标注器**（见下） |
| 聚合推理 | 点待推理图片之一 → 目标检测（固定用图片推理那套）→ 人脸识别（姓名贴到 head 框）→ 姿势检测与链式贪心绑定，原图叠 head 框 + 胶囊筛选 + 明细表（见「聚合推理」） |

### 人数折线图（视频推理）
每段视频检测时逐帧记录 `head` 类检出数，结果视频下方自动画出**人数折线图**（像剪辑时间轴）：

- **横向滚动**：滚轮直接横向平移、或按住鼠标左键拖动；滚到两端时自动交还页面滚动
- **Ctrl + 滚轮**：缩放「每帧宽度」，以鼠标位置为锚点（视口不漂移）
- **X 轴显示时间**（按视频 fps 由帧号换算），**悬停**显示该帧时间与人数
- **点击曲线跳转播放时间**：按最近采样帧定位；截取片段自动扣除起始时间，拖动平移不触发跳转。
- **人数统计**：最高、最低和全局平均人数（保留两位小数），按本次推理区间全部实际采样帧计算，不使用绘图降采样后的点。
- 数据存在 `data/outputs/frames/<结果视频名>.json`（`head_counts` + `fps` + `start_frame`）；点数过多时前端按「桶内取最大值」降采样，保住峰值
- 结果库的「预览」会同时显示结果视频 + 折线图

结果页的标注**由前端渲染**（后端只返回坐标数据），所以能交互：

- **显示方式可切「框 / 实心 / 中心点」**：实心模式给框加完全不透明填充色，密集场景看得清；中心点模式只画一个点，鼠标移上去才弹标签（并浮现它的虚线框）
- **去重**（后处理，不重跑模型，**默认不勾选**）：同类别之间，中心点距离 ≤ N 像素、或一个框被另一个覆盖 ≥ X% 时视为同一目标，只留一个；可选择保留**得分高 / 大框（外包）/ 小框（内嵌）**
- **推理尺寸 imgsz 可调**（320~1920，自动对齐 32 倍数）：提高分辨率对小目标召回提升非常明显
- **置信度滑块**二次过滤、**类别胶囊**单击开关 / 双击只看这一类 —— 只重画标注，不重跑模型
- 标注 ↔ 明细表行 **双向高亮**；点标注可把对应表格行滚进视野
- 标签**统一只在鼠标移到该标注上时才显示**（hover 表格行同样会亮）；勾选「常显标签」则全图都显示
- 每个类别一个固定颜色；明细表同时列出 **中心 (cx,cy) / 宽高 / 左上右下** 三组数值
- `后端渲染图（对照）` 链接可看 ultralytics 原始出图（后端仍会保存该图）

推理接口返回的坐标（都以**原图像素**为单位，两种写法都给）：

| 字段 | 含义 |
|------|------|
| `bbox` | `[x1, y1, x2, y2]` 左上右下 |
| `bbox_cxcywh` | `[cx, cy, w, h]` 中心点 + 宽高，由 `bbox` 反算 |
| `image_width` / `image_height` | 原图尺寸（要转 YOLO 归一化标注时，除数用这个） |

## 模型模式：普通 YOLO ↔ YOLO-World（可切换）

推理页的「模型」下拉框就是开关，**按文件名自动区分**（含 `world` 的走 World）：

| 选中权重 | 模式 | 类别 | 说明 |
|---------|------|------|------|
| `yolov8s.pt` / `yolov8m.pt` | 普通 YOLO | 固定 80 类（COCO）| 类别不能改 |
| `yolo26n.pt` | 普通 YOLO | 固定 80 类（COCO）| ultralytics 新一代模型 |
| `yolov8s/m/l-worldv2.pt` | YOLO-World | **用文字随意指定** | 零样本，输入框填 `person, chair, 书本` 即可 |

默认参数：模型 `medium.pt`、置信度 `0.05`、prompt `person`
（置信度压这么低是为了不漏小目标，会在图上产生大量框，用结果页的滑块/类别筛选降噪）

想加更多权重：把 `*world*.pt` 丢进 `weight/` 就会自动被识别为 World，不需要改代码。
官方 World 权重只有 `yolov8s/m/l/x` 两代（v1 / worldv2），**没有 n 尺度**。

- 切到 World 权重时，工具栏会**自动出现「prompt」输入框**（留空 = 用权重内置类别）
- 切换类别只需重新编码文字（约 2~3 秒），模型不用重载；相同类别会跳过重算
- World 首次使用需要 CLIP 文本编码器（约 **337MB**，在 `weight/clip/`），已装好
- 权重目录只有一个 `weight/`：后端只扫 `weight/*.pt` 生成下拉框（不递归），CLIP 缓存在子目录 `weight/clip/` 里不会混淆；路径按脚本位置解析，与启动目录无关
- 训练页 / 标注器的权重下拉**与推理一致**（含 World 权重）——World 权重适合在标注器里做零样本预标注，正式训练时建议换普通权重

依赖（已安装，重装时用）：
```powershell
.\.venv\Scripts\python.exe -m pip install "git+https://github.com/ultralytics/CLIP.git"
```

## 手动启动

```powershell
.\.venv\Scripts\python.exe app.py
```

## 训练数据集准备

放在 `data/datasets/<名称>/`，两种结构都支持：

```
data/datasets/mydata/
├── data.yaml            # 类别定义（唯一来源，由界面维护，见下）
├── images/train/*.jpg   # 训练图片
├── labels/train/*.txt   # YOLO 格式标注：class_id cx cy w h（归一化）
└── images/val/ labels/val/   # 可选，没有则用 train 当验证集
```

类别只写在 `data.yaml` 的 `names` 里（**没有 `classes.txt`**）。
文件缺失时会按目录结构 + label 自动生成；`nc` 与 `names` 不一致时会自动修好
（否则 ultralytics 会直接报 `SyntaxError` 拒绝训练）。

## 内置标注器（训练页 → 标注数据）

训练页底部「标注数据」区：选数据集 + **Train / Val** 子集 → 点缩略图进入**全屏标注器**。

| 操作 | 说明 |
|------|------|
| **预标注** | 用与推理**同一套权重选项**（含 YOLO-World + prompt）对当前这一张图做推理，结果直接变成可编辑的标注框 |
| **显示形式** | 「框 / 中心点」两种，与视频推理一致（点模式适合密集小目标） |
| **类别筛选** | 左侧类别胶囊，单击只看这一类 |
| **改框** | 拖框体移动、拖四角缩放 |
| **画新框** | 在图片空白处拖拽直接画 |
| **改类别名** | 框标签下拉的「✎ 重命名本类…」，或左侧类别旁的 `✎` |
| **新增类别** | 左侧类别列表底部的 **「＋ 新增类别」** |
| **删除** | 框右上角 `×`；或先点中框再按 `Delete`；或「清空」全部 |
| **保存** | 写回 `<数据集>/labels/<split>/<图名>.txt`（YOLO 归一化格式） |

- **类别的增删改只能在界面上做**：改完由后端直接写进 `data.yaml` 的 `names` + `nc`（两者永远一起写，不可能不一致），并清掉训练用的标注缓存
- 新增类别只**追加到末尾**，`class_id` 顺延 → 已有 label 文件无需重写、不会错位
- 进入时自动载入该图**已有标注**（若 label 文件已存在），不是每次从零开始
- 预标注检出**数据集里没有的类别**时，会自动追加成新类别（同一张图多个同类目标会去重）
- 有未保存改动时按钮显示「保存 *」，关闭会提醒
- 快捷键：`Esc` 关闭标注器，`Delete` 删除当前选中的那个标注

## 训练起点权重

训练页「起始权重」下拉框与推理页**用的是同一份清单**（`weight/*.pt` 全部），
包括 YOLO-World 权重。

> 预训练权重与你的标注没有直接绑定关系：它只提供通用视觉特征，训练时分类头会按你数据集的
> 类别数（`nc`）自动重建。**标注质量决定微调效果，预训练权重决定起点好坏**。
> 若需要 COCO 类别的开放词表能力，可先选 World 权重做预标注，再换成普通权重训练。

## 目录结构

```
watch/
├── app.py            # 后端入口：组装路由、异常处理、静态文件
├── web/              # 按功能划分的 FastAPI APIRouter
│   ├── common.py     # 请求校验、上传与批量删除辅助函数
│   ├── status.py     # 模型环境状态
│   ├── media.py      # 上传、图库、媒体文件访问
│   ├── history.py    # 推理记录与合并
│   ├── detection.py  # 图片/视频推理接口
│   ├── datasets.py   # 数据集与标注接口
│   └── jobs.py       # 训练与后台任务接口
├── project_paths.py  # 共享目录和文件格式定义
├── dataset_store.py # 数据集类别、YOLO 标注读写及缓存失效
├── vision.py         # YOLOv8 封装（模型缓存、推理、训练任务）
├── inference_history.py # 推理记录持久化、快照与合并规则
├── rtdetrv4.py       # RT-DETRv4 模型适配器
├── head_fusion.py    # medium 0.45 + RT-DETRv4 0.25 一键融合（640）
├── frontend/         # 前端：纯 HTML + CSS + JS（无构建步骤）
│   ├── index.html
│   ├── app.js        # ES module 入口，连接导航并启动加载
│   ├── js/           # core/models/uploads/preview/navigation
│   │                 # images/viewer/video/charts/outputs/training/annotator/jobs
│   ├── style.css     # 按原始顺序加载样式
│   └── styles/       # base/results/controls/annotator
├── weight/           # 唯一的权重目录（*.pt + clip/ 文本编码器缓存）
├── data/
│   ├── uploads/      # 上传的图片
│   ├── videos/       # 上传的视频
│   ├── results/      # 推理结果图
│   ├── inferences/   # 每次图片推理的独立记录与快照
│   ├── outputs/      # 推理结果视频
│   ├── datasets/     # 训练数据集
│   └── runs/         # 训练输出（含 best.pt）
├── start.bat
└── requirements.txt
```

前端使用浏览器原生 ES modules，通过 `import/export` 显式声明依赖，无需 npm、打包器或编译步骤。
`app.js` 只负责启动，各功能在 `frontend/js/` 中维护；新增页面功能从对应模块扩展。
导航在入口注入加载函数，结果查看器用回调连接记录操作，避免模块互相导入。
图片记录的选择、筛选和合并状态由 `images.js` 管理；标注会话由 `annotator.js` 管理。
后端路由调用现有推理与存储层，`vision.py` 保留原有公开函数入口，数据集读写实现转到 `dataset_store.py`。
样式按 base → results → controls → annotator 的顺序加载，保持原有覆盖顺序。

前端修改后用 `Ctrl+F5` 强制刷新，以一并更新被导入的 JS/CSS 文件缓存。启动方式仍为 `start.bat`。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | /api/yolo/status | YOLO 环境状态（GPU、可用模型、World 权重、clip 依赖） |
| POST | /api/upload | 上传图片（字段 file） |
| POST | /api/upload_video | 上传视频（字段 file） |
| GET  | /api/images | 图片 + 结果图列表 |
| GET  | /api/videos | 视频 + 结果视频列表 |
| POST | /api/detect/image/{filename} | 推理图片，参数 `conf` / `weights` / `classes`（classes 仅 World 生效） |
| POST | /api/detect/video/{filename} | 推理视频（后台任务），参数 `conf` / `stride` / `weights` / `classes` |
| GET  | /api/datasets | 数据集列表 |
| GET  | /api/datasets/{name} | 数据集详情（Train/Val 子集 + 图片清单 + 已标注数 + 类别） |
| GET  | /api/annotate/image/{dataset}/{split}/{img} | 取该图现有标注（YOLO 坐标已还原为原图像素） |
| GET  | /api/annotate/raw/{dataset}/{split}/{img} | 标注器里的原图 |
| PUT  | /api/datasets/{name}/classes | 改写类别清单（body: `{classes:[...]}`，写入 data.yaml 的 names+nc） |
| POST | /api/annotate/prelabel/{dataset}/{split}/{img} | 预标注，参数 `conf` / `weights` / `classes` |
| POST | /api/annotate/save/{dataset}/{split}/{img} | 保存标注（body: `{detections:[{class_id,bbox}]}`） |
| POST | /api/train | 启动训练，参数 dataset / epochs / imgsz / batch / lr0 / weights |
| GET  | /api/jobs/{job_id} | 后台任务进度 |
| GET  | /images/·/results/·/videos/·/outputs/ | 文件访问 |
| GET  | /docs | 接口文档 |

推理返回里带 `mode`（`yolo` / `world`）、`weight`、`classes_used`（当前生效的类别清单），前端据此显示模式提示。
`conf` 会被后端钳制到 `0.01~0.99`（ultralytics 对越界值会直接抛错）。


