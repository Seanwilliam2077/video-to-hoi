# Track 1 设计

从一段静止的单目第三人称 RGB 视频，恢复米制、统一坐标系下的人体（含手）、物体网格和逐帧物体 6D 位姿。本文记录架构、取舍和待确认的问题。结论只写依据，依据变了就改这里。

最后更新：2026-09-26。第 6 节汇总官方提交规则，含主办方 2026-09-26 的书面答复；第 5.1 节是本地评分器据此执行的规则。

## 1. 约束

- **输入**：每条视频 12–29 秒，1536×1152，30 fps，相机在单条视频内静止。公开集 30 条，10 个物体各 3 条。元数据只有物体名、一句外观描述、动作说明和物理相机名。
- **提交**：人体（含手，MHR 参数轨迹）、每个物体一份网格（同一物体的 3 条共用）、逐帧物体位姿（被挡住的帧也要有）、米制尺度和统一世界系。提交物由官方 `eval_reconstruction.py` 生成（未公开），交到 Kaggle。见第 6 节。
- **评分**：相对多目（MV）重建。5 个指标各是一个 Kaggle 比赛，单位 cm，越低越好：精度轴 CD-H、CD-O，物理轴 ACC-H、ACC-O、PEN。两轴各占一半；5 个数怎么合成总排名未公布。
- **时间**：排行榜 2026-11-04 17:00 EST 冻结。每周最多 5 次提交，最后 3 天不限。
- **算力**：本机 RTX 3080 10GB。官方工具链按 48GB（A6000 / L40S）设计，SAM 3D Objects 和 CARI4D 需要租 48GB 的 Linux 机器。本机只做胶水代码、评分和可视化。

## 2. 三个改变设计的事实

### 2.1 有带真值的开发集

Track 2 Tier 1（`track_2/tier_1_multiview_caption`）是同一个 FORM-HOI 棚的另外 30 条序列，每条带一条外视角视频（同为 1536×1152）和：

- 逐帧 SOMA-X 人体参数（77 个关节的局部旋转、根平移、MHR 身份系数、尺度参数）；
- `world_T_object`（`[x, y, z, qw, qx, qy, qz]`，OpenCV 世界系）和逐帧可见性；
- 29 个物体的米制带纹理网格；
- 每条序列的地面平面。

没有相机内外参。

与 Track 1 的关系：

- 同一套 4 台 Hawk 立体相机，同一批道具（round table、white desk、skinny wood chair、tall bar stool）。Tier 1 没写外视角是哪台相机，可按背景对上。
- Track 1 的 10 个物体里，`iron`、`big_red_bowl`、`white_desk` 在 Tier 1 有 MV 网格。Tier 1 ep10 和 Track 1 ep11 是同一个动作（桌子放倒）；Tier 1 ep7 和 Track 1 ep6 都是熨斗，同一天录，相差 4 分钟。
- Tier 2（`track_2/tier_2_synthetic_noise`）= Tier 1 + "按 Track 1 误差分布采样"的抖动、丢帧和接触误差，网格不变。

用法：Tier 1 / Tier 2 只用于本地评分器的自检。主办方已写明，重建物体和估计参数只能用 Track 1 给出的数据，不能把 Track 2 的网格、位姿或由此反推的相机参数带进 Track 1。见第 6 节。

### 2.2 视频层已有现成实现：CARI4D

工具链 `reconstruction/modules/v2d_cari4d` 的流程：

1. MoGe2 批量度量深度；
2. SAM 3D Body 初始化人体（MHR）；
3. 单目深度的尺度对齐到人体；
4. FoundationPose 首帧注册，之后逐帧跟踪；
5. CoCoNet 联合修正人和物体（在 2126 条序列上训练）；
6. 300 步接触引导优化，带时间项和人体姿态先验。

它是 Track 2 Tier 3 的参考方案。输入是视频、人和物体的 mask（H5）、**米制**物体网格。README 写明 SAM 3D 生成的网格要先自己定尺度，CARI4D 只重新居中和定向，不改尺度。

### 2.3 官方评测代码透露的指标算法

`v2d_mv_postprocess/lib/mv_eval_chamfer.py`：每帧取预测网格中在相机里可见、且落在 mask 内的顶点，算它们到立体深度点云的平均最近距离，在相机装置的世界系里算。Track 1 的 `eval_reconstruction.py` 还没公开，推测思路相同。

## 3. 架构

```
objects/<物体>/          一次性：候选网格、选中的网格、合并后的尺度、对称信息
cameras/<物理相机>/      合并后的内参、重力方向
episodes/<编号>/         Track 1：深度、mask、人体、位姿、导出
dev/tier1/<编号>/        Tier 1：和 episodes/ 同构，用来打分
scores/<运行>/           每次运行的分数和 git 哈希
```

内部表示统一放在**相机系**：人体存 MHR / SOMA-X 参数和关节，物体存网格 id、`T_cam_obj`、可见性和置信度。世界变换只在导出时乘一个 4×4，评测约定变了只改导出。

断点重跑按**输入哈希**标记，不用"输出存在就跳过"：改了网格尺度或参数后，后者会让下游悄悄复用旧结果。CARI4D 的 `.stages` 就是这么做的。

### 3.1 自己写的部分

CARI4D 不管的部分：

| 模块 | 做什么 |
|---|---|
| 相机 | 按物理相机合并所有片段的内参估计后锁定；GeoCalib（工具链 `v2d_geocalib`）给重力方向；背景像素取时间中位数得到静态深度，拟合地面和桌面 |
| mask | GroundingDINO 检测 → SAM2 传播（和工具链 MV 管线一致）；同一物体 3 条共用写得最好的描述；遮挡断了就重新检测 |
| 物体网格 | 多候选生成 + 重投影选择，见 3.3 |
| 尺度 | 人体是唯一米制锚，见 3.2 |
| 钩子 | FoundationPose 前后的筛选：去人后投影重叠、旋转跳变、对称锁定、倒着跟。Tier 1 上证明有用才加 |
| 时间 | 静止段检测并锁定位姿；平滑强度按 Tier 1 分数调（起点：工具链 `run_ekf_smoothing`） |
| 导出 | 相机系 → 提交格式和世界系 |
| 评分 | `v2hoi.score`，见第 5 节 |

### 3.2 尺度：只有一个锚

工具链的 `run_estimate_mesh_scale` 在单帧上对 0.5–2.0 倍做网格搜索，按剪影 IoU 和深度残差打分。它估出的尺度只和喂给它的深度一样"米制"。

如果网格尺度来自 UniDepth / MoGe 的原始深度，而 CARI4D 把深度尺度对齐到人体，两个锚不一致，物体会被放到和手不同的深度上，接触、穿透和 Chamfer 一起坏。

做法：

1. 每条视频：深度尺度对齐到人体（CARI4D 第 3 步）；
2. 在对齐后的深度上，选多帧无遮挡的物体点云估网格尺度；
3. 同一物体 3 条视频的估计取中位数，得到一个尺度；
4. 用"脚在地面上""物体放在桌面上"做校验。

### 3.3 物体网格

从 3 条视频里取 5–10 个候选帧，分别用 SAM 3D Objects 和 Hunyuan3D-2 生成。把每个候选放进跟踪结果，按多帧剪影 IoU 和深度残差打分，挑最好的。Tier 1 上的 iron、bowl、desk 能直接量出生成网格的形状误差。

按物体特性处理：

- **旋转对称**：bowl（连续对称）、pink foam roll（圆柱）、paint roller（滚筒会自转，位姿跟手柄）。要把对称信息给 FoundationPose，否则朝向乱跳。
- **细环**：hula hoop 单图生成基本不可用，用圆环参数（大半径、管径），从椭圆拟合加深度求。
- **白色无纹理大件**：white desk、white laptop cart，靠 mask 和深度对齐。

### 3.4 人体

一个模型走到底：SAM 3D Body（MHR）→ `v2d_sam3d_body/lib/export_soma.py` → SOMA-X。不用 GVHMR 替换全局轨迹：它会把 SMPL 的根节点定义和尺度混进来，而静止相机下根轨迹能从深度直接观测。Tier 1 证明根轨迹是瓶颈再重新考虑。

SOMA-X 的 77 个关节里有 48 个是手指关节（不含两个手腕）。如果关节加速度把手指也算进去，SAM 3D Body 的手指抖动会是主要失分项，手要单独平滑。

### 3.5 物理项

- 加速度只看提交轨迹自己的二阶差分，衡量平滑，不和多目参考相减。平滑会直接降低这一项；平滑过头则把物体拉离正确位姿，Chamfer 变差。
- 穿透交给联合优化（CARI4D 接触项 + 地面不可穿透 + 桌上物体贴桌面），不做事后推开，否则会改物体轨迹、伤到物体加速度和 Chamfer。
- 静止段（拿起前、放下后、坐凳子时）锁定位姿：抖动归零、Chamfer 更稳，实现便宜。

## 4. 片段分组

| 组 | Track 1 片段 | 主要难点 |
|---|---|---|
| 手持小物 | 3–5 碗、6–8 熨斗、12–14 锅、15–17 泡沫块、18–20 刷子、21–23 泡沫滚筒 | 手遮挡、对称 |
| 大件家具 | 9–11 桌子、27–29 小推车 | 无纹理、出画、用脚推 |
| 身体接触 | 24、26 坐凳，25 搬凳 | 凳子被人挡住，接触面大 |
| 细环 | 0–2 呼啦圈 | 细、反光、身体穿过、动作快 |

物理相机：back 5 条、front 9 条、left 11 条、right 5 条。ep12–29 都是 2026-09-10、09-11 两天录的，同一台相机的外参大概率不变。

起步：Track 1 的 ep16（泡沫块）、ep12（锅），同时跑 Tier 1 的 ep7（熨斗）、ep9（碗）。Tier 1 的每条跑两遍，一遍用真值网格、一遍用生成网格，把误差拆成网格、尺度和跟踪三部分。

## 5. 本地评分器

`python -m v2hoi.score --gt <tier1 根目录> --pred <预测根目录>`

预测根目录和 Tier 1 同构（LeRobot v2.1）：`data/chunk-000/episode_XXXXXX.parquet`（Tier 1 的列）加 `mesh/<物体>/<物体>.glb`。Tier 2 根目录可以直接当预测。管线导出也写这个格式，因为它是目前对 Track 1 提交格式的最好猜测。

人体用 SOMA-X（`py-soma-x==0.2.1`，资产版本 `466879a8`，和工具链的 `setup_soma_assets.py` 一致）正向计算得到关节和网格。这是本地评分器的实现，因为 Tier 1 参考是 SOMA-X。官方提交要的是 MHR 轨迹，不是 SOMA-X，也不是逐帧网格：管线内部保留 MHR，本地评分前用工具链的 `export_soma.py` 转成 SOMA-X。

对齐默认按官方规则（`--align first`）：用第一帧做一次 Sim(3)，套到整段。官方没说第一帧用哪些点来拟合，本地用第一帧的身体关节（不含手指），这一点是我们的假设。`--align se3 / sim3` 用全段身体关节拟合，`none` 不对齐，都只作诊断。全段拟合会把漂移摊到各帧、看起来更好，第一帧对齐不会。

| 指标 | 定义 |
|---|---|
| `human.chamfer_mm` | 每帧预测与真值 SOMA 顶点的对称 Chamfer，帧平均 |
| `human.mpjpe_mm` | 对齐后关节位置误差（诊断） |
| `human.accel_mm_f2` | 官方口径：只对预测关节轨迹做二阶差分，mm/frame²；另分身体和手指 |
| `human.accel_ref_mm_f2` | 同一量在参考轨迹上的值，用来判断平滑是否过头 |
| `human.accel_err_mm_f2` | 诊断：预测与参考二阶差分之差 |
| `object.chamfer_mm` | 两边都可见的帧上，摆好位姿的表面采样点对称 Chamfer（官方同样是世界系里摆好位姿的网格） |
| `object.shape_chamfer_mm` | 在摆好位姿的基础上再做刚体 ICP 的残差，只看形状（诊断） |
| `object.coverage` | 真值可见帧里预测给出位姿的比例。合法提交应为 1 |
| `object.accel_mm_f2` / `ang_accel_deg_f2` | 官方口径：只对预测轨迹做二阶差分，全段，包括被挡住的帧。平移用网格质心，与网格原点的选取无关 |
| `object.accel_ref_mm_f2` / `ang_accel_ref_deg_f2` | 同一量在参考轨迹上的值 |
| `object.accel_err_mm_f2` / `ang_accel_err_deg_f2` | 诊断：预测与参考之差 |
| `contact.penetration_mm` | 每帧人体顶点进入物体的最大深度，报预测、真值和两者之差；预测深度按对齐尺度换算 |
| `contact.ground_penetration_mm` | 人体和物体低于真值地面的最大深度（诊断） |

对齐和平滑两项已按主办方答复实现；Chamfer 和穿透的具体算法仍是近似，假设列在 6.3。官方脚本公开后，按它改定义，不改接口。

坐标约定（已在 Tier 1 上核对）：SOMA 人体在 Y 朝上的世界里，物体位姿和地面在 OpenCV 世界（Y 朝下、Z 朝前）。人体乘 `diag(1, -1, -1)` 后，ep7 的手腕到熨斗约 0.15 m，脚在地面上方 2–7 cm；不翻转则都差几米。Tier 1 的世界原点在某台相机附近（物体 z≈3 m，地面 y≈1.39 m）。

### 5.1 评分规则

`v2hoi.score` 按第 6 节把一次评分分成三层。

**排行榜数。** 报告里的 `leaderboard`，5 个键和 Kaggle 比赛一一对应，单位 cm，30 条按条平均：

| 键 | 排行榜列 | 本地来源 |
|---|---|---|
| `cd_h_cm` | CD-H | `human.chamfer_mm` / 10 |
| `cd_o_cm` | CD-O | `object.chamfer_mm` / 10 |
| `acc_h_cm` | ACC-H | `human.accel_mm_f2` / 10 |
| `acc_o_cm` | ACC-O | `object.accel_mm_f2` / 10 |
| `interpenetration_cm` | PEN | `contact.penetration_err_mm` / 10（代理，官方比赛未上线） |

表格前 5 列就是它们，其余都是诊断。

**官方设置**（`submission.official_settings`）：`--align first`、`--stride 1`、评全部参考片段。不满足的项列在 `submission.deviations`，这时分数只作诊断。

**合法提交**（`submission.valid`）：预测本身是否符合提交规则。不符合的列在 `submission.violations`，分数照算：

- 每个物体每帧都有位姿。预测里的 `visible` 列不参与评分，零四元数或 NaN 算缺帧。缺帧在 CD-O 和 ACC-O 里被跳过，分数会偏乐观。
- 物体网格是米制：包围盒最长边在 2 cm 到 3 m 之间。
- 物体网格不是参考网格的逐字节拷贝（不能用 Track 2 资产）。只查原样拷贝。

直接报错、不出分：预测缺参考片段（用 `--episodes` 指定子集时除外）、帧数和参考不同、人体参数有 NaN 或 inf。

`official_settings` 和 `valid` 都成立时，排行榜数才和 Kaggle 可比，仍受 6.3 那些假设的限制。`--strict` 在两者没有都成立时退出码为 1，交 Kaggle 前用它把关。README 的两条自检都会显示不是合法提交，因为它们的网格就是参考网格。

### 5.2 参照：Tier 2 对 Tier 1

主办方按"Track 1 误差分布"加的噪声，30 条全跑，默认的第一帧 Sim(3) 对齐，均值：

| 指标 | Tier 2 | 参考（Tier 1 自身） |
|---|---|---|
| 人体 Chamfer | 19.9 mm | — |
| 人体关节加速度（只看轨迹本身） | 3.5 mm/frame² | 1.9 mm/frame² |
| 物体 Chamfer | 29.5 mm | — |
| 物体加速度 / 角加速度（只看轨迹本身） | 1.8 mm/frame² / 1.6 deg/frame² | 0.7 mm/frame² / 0.4 deg/frame² |
| 穿透差 | 11.7 mm | — |
| 物体低于地面 | 26.7 mm | 2.1 mm |

报告在 `scores/tier2_vs_tier1_align-first.json`（不入库，按 README 的命令可复现）。

几点观察：

- **按 5.1 的规则，Tier 2 不是合法提交。** 网格都是参考网格；ep11 的最后一帧（900）没有物体位姿。Tier 1 参考在这一帧也是空的，这也是 30 条里唯一不可见的一帧。
- **Tier 2 第一帧没有噪声。** 30 条的第一帧 Sim(3) 都是单位变换（尺度 1 ± 3e-7），噪声从第一帧之后才累积，所以这里和不对齐的结果几乎一样（人体 / 物体 Chamfer 19.9 / 29.4 mm）。它测不出第一帧出错的代价，而官方对齐恰恰对第一帧最敏感。
- **多目参考本身也不是静止的。** 参考的人体加速度约 1.9 mm/frame²，物体约 0.7 mm/frame²。平滑压到明显低于这个量级，多半是在抹掉真实运动，Chamfer 会付出代价。Tier 2 大约是参考的 2–4 倍。

- **对齐方式影响很大。** ep7 上按人体做 SE(3) 对齐，人体 Chamfer 从 24.6 降到 10.5 mm，物体 Chamfer 却从 12.8 升到 18.5 mm：Tier 2 的人体噪声里有物体不共享的整体偏移。官方现在定为第一帧 Sim(3)，见第 6 节。
- **穿透是最不可靠的指标。** Tier 1 网格都不是水密的，符号距离靠近邻采样点法向投票近似。round table（ep14）的真值穿透算出 143 mm，基本是假象；其余真值在 0–29 mm。之后换成广义环绕数（generalized winding number）再看。
- **形状分对大旋转误差敏感。** 同一个网格，Tier 2 上仍有 0–5.5 mm（碗、篮子、圆桌这类对称物体），因为 ICP 从带噪声的朝向出发会停在局部极小。只把它当诊断看。

## 6. 官方提交

依据：挑战页面（工具链 `docs/v2d_challenge/index.html`，v0.4.0，2026-09-23）；排行榜聚合代码（`github.com/MVerghese/v2d-leaderboard-data` 的 `aggregator/tracks.json`、`v2d_aggregate.py`，页面的排行榜从这里取数）；Track 1 数据集 README；主办方 2026-09-26 的书面答复（6.4）。截至 2026-09-26，`eval_reconstruction.py` 没有公开，Track 1 的 Kaggle 比赛页面也打不开（404）。

### 6.1 交什么、交到哪

| 项 | 已确认 | 来源 |
|---|---|---|
| 平台 | Kaggle。Track 1 拆成 5 个比赛，每个只有一个分数：`v2d-challenge-track1-cd-h`、`-cd-o`、`-acc-h`、`-acc-o`、`-pen` | 聚合代码 |
| 提交物 | 用官方 `eval_reconstruction.py` 在测试集上生成的文件 | 挑战页面 |
| 测试集 | Track 1 公开的 30 条视频，真值不公开 | 数据集 README、挑战页面 |
| 人体 | MHR 参数轨迹，含手。MV 参考本身就是 MHR（工具链 `v2d_sam3d_body`：`global_rot`、`body_pose_params`、`hand_pose_params`、`scale_params`、`shape_params` 加平移）；Track 2 的 SOMA-X 是由它转出来的 | 主办方答复、工具链 |
| 物体 | 每个物体一份米制网格；逐帧 6D 位姿，被挡住的帧也要有，轨迹连续 | 挑战页面、主办方答复 |
| 坐标 | 米制、统一世界系。评测时用参考轨迹的第一帧做一次 Sim(3)，作用到整段 | 挑战页面、主办方答复 |
| 数据 | 只用 Track 1 的数据。不用 Track 2 的网格、位姿，也不用由它们反推的相机参数 | 主办方答复 |
| 次数 | 每周 5 次，最后 3 天不限；2026-11-04 17:00 EST 冻结。Kaggle 上立刻出分，挑战页面最多晚 24 小时 | 挑战页面 |

### 6.2 怎么评

| 排行榜列 | 轴 | 已知定义 |
|---|---|---|
| CD-H (cm) | 精度 | 对 MV 人体网格的 Chamfer |
| CD-O (cm) | 精度 | 世界系里摆好位姿的物体网格，对 MV 物体网格的 Chamfer。位姿误差算在里面 |
| ACC-H (cm) | 物理 | 只对预测关节轨迹做二阶差分，衡量平滑，不减参考 |
| ACC-O (cm) | 物理 | 只对预测物体轨迹做二阶差分 |
| PEN (cm) | 物理 | 人–物穿透，页面写"compared to MV"。比赛还没上线 |

全部越低越好。两轴各占一半用于评奖，赛道冠军是"track leaderboard 第一"。每个 Kaggle 比赛按队伍最好的一次计分。挑战页面把 5 个比赛按 Kaggle 用户名拼成一张表，默认按 CD-H 排序，可以按任一列重排；缺某一列的队伍标为 incomplete。

页面上写的是"acceleration error compared to MV"，主办方答复是"只看预测轨迹自身"。以更晚、更具体的答复为准。

### 6.3 还没公布的，和本地的假设

| 问题 | 本地做法 |
|---|---|
| 提交文件格式：MHR 怎么存、网格和位姿用什么格式 | 本地用 Tier 1 格式（第 5 节）。脚本公开后只改导出 |
| 第一帧 Sim(3) 用哪些点拟合 | 第一帧身体关节，不含手指 |
| ACC 标的单位是 cm：是 cm/frame² 还是换算到秒 | cm/frame²，30 fps |
| ACC-H 用哪些关节、取什么统计量 | 77 个 SOMA-X 关节逐帧加速度模长的平均；另报身体和手指 |
| ACC-O 取物体上哪个点，算不算转动 | 网格质心的平动；角加速度只作诊断 |
| CD 在哪些帧、用顶点还是表面采样、只看可见部分还是全部 | 人体：每帧全部顶点；物体：参考可见的帧，表面采样 1 万点 |
| PEN 的定义 | 每帧人体顶点进入物体的最大深度，取预测和参考之差的绝对值 |
| 30 条怎么汇总：按条还是按帧 | 按条平均 |
| 5 个数怎么合成排名，轴内怎么合成 | 不合成，只报 5 个数 |
| 每周 5 次是 5 个比赛合计还是每个比赛单独算 | 按合计规划 |

### 6.4 主办方答复

2026-09-26，主办方回复 Zijun。原文如下，随后是对管线的约束。

> The eval script does Sim(3) alignment of submitted results with the first frame of the reference trajectory.
> Please submit human trajectories in MHR representation.
> Object chamfer distance is computed from the posed mesh in world frame.
> Acceleration error is computed as the second order difference from predicted trajectories alone. It measures the smoothness of trajectories.
> Yes, please provide a continuous trajectory for each object, including frames where it may be occluded.
> Are you referring to Tier 1 assets for Track 2? Please do not use the assets from Track 2. Instead, reconstruct objects and estimate parameters only from data provided by Track 1. We will clarify this for other participants as well.

| 问题 | 决定 |
|---|---|
| 对齐 | 评测脚本用参考轨迹的第一帧做一次 Sim(3)，作用到整段。不再逐帧对齐 |
| 人体 | 提交 MHR 参数轨迹 |
| 物体 Chamfer | 世界系里、已经摆好位姿的网格。位姿误差算进这个分数 |
| 加速度 | 只对提交轨迹做二阶差分，衡量平滑，不减去多目参考 |
| 遮挡 | 每个物体都要连续轨迹，被挡住的帧也要有位姿 |
| Track 2 | 不用 Track 2 的网格、位姿，也不用它们反推的相机参数。物体和参数只从 Track 1 的数据估计 |

因此：第一帧的人和物体必须可靠，序列内部不能有尺度漂移。遮挡处要插值或继续跟踪，不能缺帧，也不能填全 0。平滑能降低加速度分，但会伤害 Chamfer，两边一起看。Tier 1 可以留在本地评分器里做自检，不能进入重建或提交。

## 7. 时间线

| 周 | 截止 | 内容 |
|---|---|---|
| 1 | 10-01 | 发问题、申请门控权重（`facebook/sam-3d-body-dinov3`、`facebook/sam-3d-objects`、`nvidia/cari4d_commercial`）、准备算力；评分器，用 Tier 2 对 Tier 1 验证 |
| 2 | 10-08 | CARI4D 原样跑 ep16、ep12 和 Tier 1 的 ep7、ep9；尽早交一次，确认格式 |
| 3 | 10-15 | 物体层（多候选 + 尺度合并）、物理相机内参、世界系；30 条全部有输出后交完整版本 |
| 4–5 | 10-29 | 静止段、对称、遮挡重注册、坐和脚推的接触、呼啦圈 |
| 6 | 11-04 | 在 Tier 1 上调平滑和联合优化权重；最后 3 天集中提交 |
