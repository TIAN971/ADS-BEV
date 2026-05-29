# DriveLM-v1 数据集使用指南

## 1. 数据集用途

本项目使用 **DriveLM-v1 / DriveLM-nuScenes** 作为语言驾驶任务数据集，用于后续实现 **语言引导的 BEV 感知注意力增强模块**。

DriveLM-v1 在本项目中的主要作用是提供自然语言驾驶场景信息，包括场景描述、关键帧、多摄像头图像路径、关键目标信息以及问答标注。后续模型会利用这些自然语言信息生成 BEV 空间注意力掩码，从而增强指令相关区域的感知特征。

本项目中的数据流关系如下：

```text
DriveLM-v1 自然语言问答 / 场景描述
        ↓
关键帧 frame token 与图像路径
        ↓
关键目标与 2D bbox 信息
        ↓
语言指令相关区域定位
        ↓
语言引导的 BEV attention mask
        ↓
增强 BEVFormer 的 BEV 感知特征
```

---

## 2. 数据文件组成

本项目当前使用两类数据：

### 2.1 DriveLM 标注文件

文件名：

```text
v1_0_train_nus.json
```

建议存放位置：

```text
~/projects/ads_bev_project/data/drivelm/QA_dataset_nus/v1_0_train_nus.json
```

该文件是 DriveLM-nuScenes 的核心标注文件，包含以下内容：

```text
1. 场景 scene 信息
2. 关键帧 key frame 信息
3. 多摄像头图像路径
4. 关键目标 key object 信息
5. perception / prediction / planning / behavior 等问答标注
```

---

### 2.2 nuScenes 数据

如果采用完整标准流程，需要准备完整的 nuScenes trainval 数据集以及 CAN bus expansion。

建议存放位置：

```text
~/projects/ads_bev_project/data/nuscenes/
```

完整目录结构应包括：

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
└── can_bus/
```

各目录含义如下：

| 目录 | 作用 |
|---|---|
| `maps/` | 地图文件 |
| `samples/` | 关键帧传感器数据，包括六个摄像头图像 |
| `sweeps/` | 非关键帧传感器数据 |
| `v1.0-trainval/` | nuScenes trainval 标注与元数据 |
| `can_bus/` | 车辆 CAN bus 数据，用于 BEVFormer 时间与位姿相关信息 |

---

## 3. 推荐项目目录结构

本项目建议采用如下目录结构：

```text
ads_bev_project/
├── BEVFormer/
├── mmdetection3d/
├── DriveLM/
├── data/
│   ├── drivelm/
│   │   ├── QA_dataset_nus/
│   │   │   └── v1_0_train_nus.json
│   │   └── processed/
│   │       ├── drivelm_frames.jsonl
│   │       ├── drivelm_qa.jsonl
│   │       └── drivelm_objects.jsonl
│   └── nuscenes/
│       ├── maps/
│       ├── samples/
│       ├── sweeps/
│       ├── v1.0-trainval/
│       └── can_bus/
├── scripts/
└── docs/
    ├── drivelm_dataset_guide.md
    ├── drivelm_summary.json
    └── baseline_report.md
```

---

## 4. 数据下载与放置说明

### 4.1 DriveLM JSON 文件

下载 `v1_0_train_nus.json` 后，将其放入：

```text
~/projects/ads_bev_project/data/drivelm/QA_dataset_nus/
```

如果文件下载到了 Windows 的 Downloads 文件夹，可以在 WSL 中复制：

```bash
mkdir -p ~/projects/ads_bev_project/data/drivelm/QA_dataset_nus

cp /mnt/c/Users/tian/Downloads/v1_0_train_nus.json \
   ~/projects/ads_bev_project/data/drivelm/QA_dataset_nus/
```

先查看用户名：

```bash
ls /mnt/c/Users
```

然后将命令中的替换成自己的 Windows 用户名。

---

### 4.2 nuScenes trainval 数据

完整标准流程需要下载：

```text
v1.0-trainval_meta
v1.0-trainval01_blobs
v1.0-trainval02_blobs
v1.0-trainval03_blobs
v1.0-trainval04_blobs
v1.0-trainval05_blobs
v1.0-trainval06_blobs
v1.0-trainval07_blobs
v1.0-trainval08_blobs
v1.0-trainval09_blobs
v1.0-trainval10_blobs
CAN bus expansion
```

这些文件解压后应统一放入：

```text
~/projects/ads_bev_project/data/nuscenes/
```

如果压缩包是 `.tgz` 格式，可以批量解压：

```bash
mkdir -p ~/projects/ads_bev_project/data/nuscenes

for f in /mnt/c/Users/tian/Downloads/v1.0-trainval*.tgz; do
    echo "Extracting $f"
    tar -xzf "$f" -C ~/projects/ads_bev_project/data/nuscenes
done
```

如果压缩包是 `.zip` 格式，可以使用：

```bash
for f in /mnt/c/Users/tian/Downloads/v1.0-trainval*.zip; do
    echo "Extracting $f"
    unzip "$f" -d ~/projects/ads_bev_project/data/nuscenes
done
```

CAN bus 数据解压：

```bash
unzip /mnt/c/Users/tian/Downloads/can_bus.zip \
    -d ~/projects/ads_bev_project/data/nuscenes
```

---

## 5. 数据目录检查

解压完成后，检查 nuScenes 目录：

```bash
ls ~/projects/ads_bev_project/data/nuscenes
```

正常情况下应看到：

```text
maps
samples
sweeps
v1.0-trainval
can_bus
```

检查摄像头数据：

```bash
ls ~/projects/ads_bev_project/data/nuscenes/samples
```

应看到类似：

```text
CAM_BACK
CAM_BACK_LEFT
CAM_BACK_RIGHT
CAM_FRONT
CAM_FRONT_LEFT
CAM_FRONT_RIGHT
LIDAR_TOP
RADAR_BACK_LEFT
RADAR_BACK_RIGHT
RADAR_FRONT
RADAR_FRONT_LEFT
RADAR_FRONT_RIGHT
```

检查元数据：

```bash
ls ~/projects/ads_bev_project/data/nuscenes/v1.0-trainval | head
```

应看到类似：

```text
sample.json
sample_data.json
scene.json
sample_annotation.json
calibrated_sensor.json
ego_pose.json
```

---

## 6. DriveLM 原始 JSON 结构说明

`v1_0_train_nus.json` 是一个嵌套结构，整体逻辑如下：

```text
scene_token
    └── key_frames
            └── frame_token
                    ├── image_paths
                    ├── key_object_infos
                    └── QA
```

### 6.1 scene_token

`scene_token` 表示一个驾驶场景。每个 scene 下包含多个关键帧。

### 6.2 key_frames

`key_frames` 表示该场景中的关键帧。每个关键帧对应一个 `frame_token`。

### 6.3 image_paths

`image_paths` 表示该帧对应的多摄像头图像路径，通常包括：

```text
CAM_FRONT
CAM_FRONT_LEFT
CAM_FRONT_RIGHT
CAM_BACK
CAM_BACK_LEFT
CAM_BACK_RIGHT
```

这些图像用于后续进行多视角输入检查和可视化。

### 6.4 key_object_infos

`key_object_infos` 表示当前帧中的关键目标信息，通常包括：

```text
目标类别
目标状态
视觉描述
2D bbox
```

这些信息可以用于定位自然语言指令中提到的目标，例如车辆、行人、交通锥、障碍物等。

### 6.5 QA

`QA` 表示该帧对应的问答标注，通常包括：

```text
perception
prediction
planning
behavior
```

这些自然语言问答数据是后续语言语义解析模块的重要输入来源。

---

## 7. 数据预处理目标

为了方便后续训练、分析和模块设计，需要将原始嵌套 JSON 拆分成三个中间文件：

```text
data/drivelm/processed/drivelm_frames.jsonl
data/drivelm/processed/drivelm_qa.jsonl
data/drivelm/processed/drivelm_objects.jsonl
```

### 7.1 drivelm_frames.jsonl

一行代表一个关键帧。

主要字段包括：

```text
scene_token
frame_token
scene_description
image_paths
num_key_objects
qa_types
```

用途：

```text
建立 frame 与多摄像头图像之间的映射关系。
```

---

### 7.2 drivelm_qa.jsonl

一行代表一个 QA 问答样本。

主要字段包括：

```text
scene_token
frame_token
qa_type
question
answer
chain
con_up
con_down
cluster
layer
```

用途：

```text
为后续语言编码器和语义注意力模块提供自然语言输入。
```

---

### 7.3 drivelm_objects.jsonl

一行代表一个关键目标。

主要字段包括：

```text
scene_token
frame_token
object_id
category
status
visual_description
bbox_2d
```

用途：

```text
建立关键目标、语言描述和图像区域之间的对应关系。
```

---

## 8. 预处理脚本说明

项目中建议创建三个脚本：

```text
scripts/
├── inspect_drivelm.py
├── preprocess_drivelm.py
└── check_drivelm_images.py
```

### 8.1 inspect_drivelm.py

作用：

```text
读取原始 DriveLM JSON，统计 scene 数量、frame 数量、QA 类型、目标类别和相机视角。
```

运行：

```bash
cd ~/projects/ads_bev_project
conda activate bevformer

python scripts/inspect_drivelm.py
```

输出：

```text
docs/drivelm_summary.json
```

---

### 8.2 preprocess_drivelm.py

作用：

```text
将原始嵌套 JSON 拆分成 frame、QA、object 三类 jsonl 文件。
```

运行：

```bash
python scripts/preprocess_drivelm.py
```

输出：

```text
data/drivelm/processed/drivelm_frames.jsonl
data/drivelm/processed/drivelm_qa.jsonl
data/drivelm/processed/drivelm_objects.jsonl
```

---

### 8.3 check_drivelm_images.py

作用：

```text
检查 DriveLM JSON 中的 image_paths 是否能在本地数据目录中找到对应图像。
```

运行：

```bash
python scripts/check_drivelm_images.py
```

理想输出：

```text
Missing images: 0
```

如果缺失图像数量不为 0，说明图像数据路径没有正确对齐，需要检查 `data/nuscenes/samples/` 是否存在，以及 JSON 中的相对路径是否能正确解析。

---

## 9. DriveLM 与 BEVFormer 的关系

DriveLM-v1 本身主要提供自然语言问答、关键目标和图像路径信息。BEVFormer baseline 则需要使用 nuScenes 的标准数据结构，包括：

```text
samples/
sweeps/
v1.0-trainval/
maps/
can_bus/
```

因此，本项目的数据关系是：

```text
DriveLM-v1
    提供自然语言指令、QA、关键目标信息

nuScenes
    提供多摄像头图像、3D 标注、车辆位姿、传感器标定

BEVFormer
    使用 nuScenes 数据生成 BEV 感知 baseline
```

后续需要建立如下映射关系：

```text
DriveLM frame_token
        ↓
nuScenes sample token
        ↓
BEVFormer 输入样本
        ↓
语言引导 BEV attention 模块
```

---

## 10. BEVFormer 数据准备

为了让 BEVFormer 读取 nuScenes 数据，需要在 BEVFormer 目录下建立数据软链接：

```bash
cd ~/projects/ads_bev_project/BEVFormer

mkdir -p data

ln -sfn ~/projects/ads_bev_project/data/nuscenes data/nuscenes
ln -sfn ~/projects/ads_bev_project/data/nuscenes/can_bus data/can_bus
```

然后生成 BEVFormer 需要的 info 文件：

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0 \
    --canbus ./data
```

成功后检查：

```bash
ls data/nuscenes | grep infos
```

应看到类似：

```text
nuscenes_infos_temporal_train.pkl
nuscenes_infos_temporal_val.pkl
```

这些文件是后续训练和评估 BEVFormer baseline 的必要输入。

---

## 11. 数据验证标准

本阶段数据准备完成的判断标准如下：

```text
1. v1_0_train_nus.json 可以正常读取
2. data/nuscenes/ 目录下包含 samples、sweeps、maps、v1.0-trainval、can_bus
3. inspect_drivelm.py 可以输出统计信息
4. preprocess_drivelm.py 可以生成三个 jsonl 文件
5. check_drivelm_images.py 显示 Missing images: 0
6. BEVFormer 可以基于 nuScenes 数据生成 info pkl 文件
```

---

## 12. 当前阶段产出

完成本阶段后，应得到以下文件：

```text
docs/
├── drivelm_dataset_guide.md
├── drivelm_summary.json
└── baseline_report.md

data/drivelm/processed/
├── drivelm_frames.jsonl
├── drivelm_qa.jsonl
└── drivelm_objects.jsonl

data/nuscenes/
├── nuscenes_infos_temporal_train.pkl
└── nuscenes_infos_temporal_val.pkl
```

这些文件将用于支持后续：

```text
1. BEVFormer baseline 复现
2. 语言引导 attention 模块设计
3. DriveLM 与 BEVFormer 数据对齐
4. 后续实验分析和报告撰写
```


## 13. 本阶段总结

本阶段完成后，项目将具备标准数据基础：

```text
DriveLM-v1 用于语言指令与场景理解
nuScenes 用于 BEVFormer 感知基线
processed JSONL 文件用于后续语言引导注意力模块
```

这一步是后续实现 **语言引导 BEV 感知注意力增强模块** 的基础。
