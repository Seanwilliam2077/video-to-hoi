# video-to-hoi

从一段单目第三人称视频恢复人和物体的三维运动（V2D Challenge Track 1）。

输入是一台静止 RGB 相机拍的视频，以及要跟踪的物体描述。输出是米制尺度、统一坐标系里的人体、物体形状和逐帧物体位姿。

架构、取舍和待确认的问题见 [docs/design.md](docs/design.md)。

## 状态

- 设计：`docs/design.md`
- Tier 1 本地评分器：`python -m v2hoi.score`，可用
- 重建管线：尚未实现

## 安装

需要 Python 3.10+，以及已装好的 CUDA 版 PyTorch。下面的 venv 复用系统里的 torch，不会再装一份 CPU 版：

```bash
uv venv .venv --python 3.10 --system-site-packages
uv pip install --python .venv/Scripts/python.exe numpy scipy pandas pyarrow trimesh huggingface_hub pytest warp-lang rtree cholespy usd-core
uv pip install --python .venv/Scripts/python.exe --no-deps "py-soma-x==0.2.1"
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
```

`py-soma-x` 第一次运行时从 Hugging Face 下载 SOMA-X 资产，版本固定在官方工具链用的那一版。

## 数据

```bash
.venv/Scripts/python.exe -m v2hoi.download
```

下载到 `data/v2d/`：Track 2 Tier 1 的真值（人体、物体位姿、网格、地面）、Tier 2 的加噪版本、Track 1 的元数据，约 400 MB。加 `--videos` 同时下载 Tier 1 和 Track 1 的视频。

## 评分

预测目录和 Tier 1 同构：`meta/info.json`、`meta/episodes_metadata.jsonl`、`data/chunk-000/episode_XXXXXX.parquet`（Tier 1 的列）和 `mesh/<物体>/<物体>.glb`。

```bash
.venv/Scripts/python.exe -m v2hoi.score --pred <预测目录>
```

常用参数：`--episodes 7 9` 只评部分片段；`--stride 3` 逐帧网格指标隔帧算；`--align se3|sim3|none` 选对齐方式（默认 SE(3)，按身体关节）；`--pred-mesh-dir` 从别处找预测网格。默认要求预测覆盖所有真值片段和所有真值可见的物体帧；`--allow-incomplete` 可用于缺帧诊断，报告会标记为不可用于排名比较。结果打印成表，完整报告写到 `scores/`。

两个自检：

```bash
.venv/Scripts/python.exe -m v2hoi.score --pred data/v2d/track_2/tier_1_multiview_caption
```

```bash
.venv/Scripts/python.exe -m v2hoi.score --pred data/v2d/track_2/tier_2_synthetic_noise --pred-mesh-dir data/v2d/track_2/tier_1_multiview_caption/mesh --align none --allow-incomplete
```

第一条真值对真值，所有误差应接近 0。第二条是主办方按 Track 1 误差分布加的噪声，可作参照。指标定义见 `docs/design.md` 第 5 节。

## 测试

```bash
.venv/Scripts/python.exe -m pytest
```
