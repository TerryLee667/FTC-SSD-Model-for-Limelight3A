# Win11 从零开始训练 SSD MobileNetV2 FPN-lite

本文档总结了在 Windows 11 上通过 WSL 2 配置完整训练环境、使用 TensorFlow Object Detection API 训练 SSD MobileNetV2 FPN-lite 320×320 模型的全部步骤，并包含了常见错误的解决方法、模型评估、INT8 量化导出以及各模型格式的规范说明。

代码完全在本地运行，不依赖notebook或google drive，可供无法直接使用官方工具的FTC队伍开发limelight3A的视觉系统。

## 注意：本教程涉及对python底层代码的侵入性改动，只适用于SSD MobileNet的开发，也请使用虚拟机而避免在底层的linux系统上操作

***

## 1. 系统要求

- **操作系统**：Windows 11（版本 22000 或更高）
- **处理器**：x64，支持虚拟化技术（已启用）
- **内存**：建议 16GB 以上
- **硬盘**：至少 50GB 可用空间（用于 WSL、数据集、模型）
- **可选**：NVIDIA GPU（用于加速训练，无 GPU 也可用 CPU）

***

## 2. 安装 WSL 2 与 Ubuntu

### 经过作者长时间的尝试和无数次失败，确认了没有任何一组可以训练SSD，兼容win11且互相兼容的python包版本，所以使用linux虚拟机环境运行

以**管理员身份**打开 PowerShell，执行以下命令：

```powershell
# 启用 WSL 和虚拟机平台
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

**重启计算机**。重启后，继续以管理员身份执行：

```powershell
# 设置 WSL 2 为默认版本
wsl --set-default-version 2

# 安装 Ubuntu 24.04
wsl --install -d Ubuntu-24.04
```

首次启动 Ubuntu 时，按提示设置用户名和密码。

更新 WSL 内核（可选）：

```powershell
wsl --update
```

***

## 3. 配置 Ubuntu 基础环境

打开 Ubuntu 终端（开始菜单或运行 `wsl`），执行：

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential git wget software-properties-common protobuf-compiler
```

### 3.1 安装 Python 3.10

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

### 3.2 由于接下来涉及侵入性改动，创建虚拟环境（作者已踩坑）

```bash
python3.10 -m venv ~/SSD_cpu_gpu
source ~/SSD_cpu_gpu/bin/activate
```

### 3.3 配置 pip 镜像源（加速下载）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip
```

### 3.4 GPU加速训练配置（可选）

如果你的电脑配有NVIDIA显卡（如RTX 30/40/50系列），可以启用GPU加速训练，显著提升训练速度。本教程已在RTX 5070 Ti（Blackwell架构）上验证通过。

#### 3.4.1 Windows宿主机：安装WSL专用NVIDIA驱动

1. 下载并安装最新的[NVIDIA WSL驱动](https://developer.nvidia.com/cuda/wsl)（版本需≥515.65.01）。
2. 安装后重启Windows。
3. 在PowerShell中运行`nvidia-smi`，确认驱动已加载且能看到GPU信息。

#### 3.4.2 WSL内：无需手动安装CUDA Toolkit

TensorFlow 2.15的pip包`tensorflow[and-cuda]==2.15.0.post1`会自动下载并安装匹配的CUDA库（如`nvidia-cublas-cu12`、`nvidia-cudnn-cu12`等）到虚拟环境中，因此**不需要**在WSL中手动安装系统级CUDA Toolkit或cuDNN。

#### 3.4.3 环境变量配置（重要）

为确保TensorFlow能找到这些动态库，在激活虚拟环境后，执行：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/SSD_cpu_gpu/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$HOME/SSD_cpu_gpu/lib/python3.10/site-packages/nvidia/cudnn/lib
```

建议将上述命令追加到`~/.bashrc`中，以便每次登录自动生效：

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/SSD_cpu_gpu/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$HOME/SSD_cpu_gpu/lib/python3.10/site-packages/nvidia/cudnn/lib' >> ~/.bashrc
source ~/.bashrc
```

#### 3.4.4 验证GPU是否可用

在虚拟环境中运行：

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

如果输出`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`，则表示GPU已成功启用。

> **注意**：对于RTX 50系列（Blackwell架构），TensorFlow 2.15没有预编译内核，首次训练时会触发PTX即时编译警告（如`TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0`），这是正常现象，编译完成后训练会正常进行，不会影响最终精度。

#### 3.4.5 常见GPU相关问题

| 现象                                               | 可能原因                      | 解决方法                                        |
| :----------------------------------------------- | :------------------------ | :------------------------------------------ |
| `nvidia-smi`显示驱动正常，但TensorFlow找不到GPU             | `LD_LIBRARY_PATH`未正确设置    | 执行3.4.3节的环境变量配置                             |
| `Failed to call cuInit: UNKNOWN ERROR`           | WSL驱动未正确加载或版本过低           | 重启Windows，重新安装最新WSL驱动                       |
| 训练时出现大量`Failed to compile generated PTX`警告       | GPU架构较新，TensorFlow需要JIT编译 | 可忽略，首次编译后会自动缓存                              |
| `Could not load dynamic library 'libcudnn.so.8'` | cuDNN库路径缺失                | 检查`LD_LIBRARY_PATH`是否包含`nvidia/cudnn/lib`目录 |

***

## 4. 安装训练所需 Python 包

```bash
# 安装 TensorFlow 2.15 GPU 版（无 GPU 时自动降级为 CPU）
pip install tensorflow[and-cuda]==2.15.0.post1

# 降级 numpy 以兼容 opencv
pip install numpy==1.23.5

# 安装其他依赖
pip install opencv-python==4.8.1.78 pillow lxml matplotlib pyyaml scipy==1.10.1
pip install tf_slim pycocotools lvis tensorflow-io==0.36.0
```

***

## 5. 安装 TensorFlow Object Detection API

### 5.1 克隆模型仓库并编译 Protobuf（这一步最好用VPN）

```bash
cd ~
git clone https://github.com/tensorflow/models.git
cd ~/models/research
protoc object_detection/protos/*.proto --python_out=.
```

### 5.2 侵入性改动：修复 TensorFlow 2.15 兼容性问题（作者已踩坑）

```bash
# 修复 freezable_sync_batch_norm.py
sed -i 's/tf.keras.layers.experimental.SyncBatchNormalization/tf.keras.layers.BatchNormalization/g' \
    ~/models/research/object_detection/core/freezable_sync_batch_norm.py
```

### 5.3 侵入性改动：跳过 EfficientNet 依赖（SSD MobileNetV2 不需要，作者已踩坑）

创建假的 `official` 模块：

```bash
mkdir -p ~/SSD_cpu_gpu/lib/python3.10/site-packages/official/modeling/optimization

cat > ~/SSD_cpu_gpu/lib/python3.10/site-packages/official/__init__.py << 'EOF'
# Fake official module
EOF

cat > ~/SSD_cpu_gpu/lib/python3.10/site-packages/official/modeling/__init__.py << 'EOF'
# Fake official.modeling
EOF

cat > ~/SSD_cpu_gpu/lib/python3.10/site-packages/official/modeling/optimization/__init__.py << 'EOF'
from . import ema_optimizer
EOF

cat > ~/SSD_cpu_gpu/lib/python3.10/site-packages/official/modeling/optimization/ema_optimizer.py << 'EOF'
class EMAOptimizer:
    def __init__(self, *args, **kwargs):
        pass
EOF
```

### 5.4 侵入性改动：修复 `ssd_efficientnet_bifpn_feature_extractor.py` 导入错误（作者已踩坑）

```bash
cd ~/models/research/object_detection/models
cp ssd_efficientnet_bifpn_feature_extractor.py ssd_efficientnet_bifpn_feature_extractor.py.bak

# 删除所有对 official.vision 的导入行
sed -i '/from official\.vision/d' ssd_efficientnet_bifpn_feature_extractor.py
sed -i '/from official\.legacy/d' ssd_efficientnet_bifpn_feature_extractor.py

# 将 try-except 块替换为 pass
python3 << 'EOF'
file_path = 'ssd_efficientnet_bifpn_feature_extractor.py'
with open(file_path, 'r') as f:
    lines = f.readlines()
new_lines = []
skip_try = False
for line in lines:
    if line.strip().startswith('try:'):
        new_lines.append('    # EfficientNet imports skipped for SSD MobileNet V2\n')
        new_lines.append('    pass\n')
        skip_try = True
        continue
    if skip_try and line.strip().startswith('except ModuleNotFoundError:'):
        skip_try = False
        continue
    if not skip_try:
        new_lines.append(line)
with open(file_path, 'w') as f:
    f.writelines(new_lines)
print("文件已修复")
EOF
```

### 5.5 侵入性改动：修复 tf\_slim 兼容性问题（作者已踩坑）

`tf_slim` 中的某些 API 在 TensorFlow 2.x 中已被移除，需要手动替换。

```bash
# 找到 tf_slim 安装路径
TF_SLIM_PATH=$(python -c "import tf_slim, os; print(os.path.dirname(tf_slim.__file__))")
echo "tf_slim 路径: $TF_SLIM_PATH"

# 备份原文件
cp $TF_SLIM_PATH/data/tfexample_decoder.py $TF_SLIM_PATH/data/tfexample_decoder.py.bak

# 替换 control_flow_ops 调用为 tf 调用
sed -i 's/control_flow_ops\.case/tf.case/g' $TF_SLIM_PATH/data/tfexample_decoder.py
sed -i 's/control_flow_ops\.cond/tf.cond/g' $TF_SLIM_PATH/data/tfexample_decoder.py
sed -i 's/control_flow_ops\.while_loop/tf.while_loop/g' $TF_SLIM_PATH/data/tfexample_decoder.py

# 确保 import tensorflow as tf 位于正确位置（在 from __future__ 之后）
nano $TF_SLIM_PATH/data/tfexample_decoder.py
# 如果有多余的行请手动删除

# 验证修复
head -10 $TF_SLIM_PATH/data/tfexample_decoder.py
```

期望输出类似：

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
...
```

### 5.6 设置 PYTHONPATH

```bash
echo 'export PYTHONPATH=$PYTHONPATH:~/models/research:~/models/research/slim' >> ~/.bashrc
source ~/.bashrc
```

### 5.7 验证环境

```bash
python -c "from object_detection.builders import model_builder; print('✅ 环境配置成功')"
```

如果没有报错，则环境配置完成。

***

## 6. 准备数据集

### 6.1 将 TFRecord 文件复制到 WSL

假设你的训练集、验证集和测试集文件（`train.tfrecord`, `eval.tfrecord`, `test.tfrecord`）在 Windows 的 `C:\data` 目录下。

```bash
mkdir -p ~/data
cp /mnt/c/data/*.tfrecord ~/data/
```

### 6.2 创建标签映射文件 `label_map.pbtxt`

根据你的数据集类别（例如 `cat`, `dog`），创建文件：

```bash
mkdir -p ~/workspace
nano ~/workspace/label_map.pbtxt
```

内容格式（id 从 1 开始，必须与 TFRecord 中的标签 id 一致）：

```
item {
    id: 1
    name: 'cat'
}
item {
    id: 2
    name: 'dog'
}
```

保存退出（`Ctrl+X`, `Y`, `Enter`）。

***

## 7. 下载预训练模型和配置文件

```bash
cd ~/models/research/object_detection
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
tar -xzvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
cp ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config ~/workspace/
```

***

## 8. 修改配置文件 `pipeline.config`

使用 `nano` 或 `vim` 编辑：

```bash
nano ~/workspace/pipeline.config
```

必须修改以下字段（使用绝对路径）：

| 字段                                  | 说明                         | 示例值                                                                                                                                   |
| :---------------------------------- | :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| `num_classes`                       | 类别数                        | `num_classes: 2`                                                                                                                      |
| `fine_tune_checkpoint`              | 预训练模型路径                    | `fine_tune_checkpoint: "/home/user/models/research/object_detection/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0"` |
| `fine_tune_checkpoint_type`         | 固定为 `"detection"`          | `fine_tune_checkpoint_type: "detection"`                                                                                              |
| `train_input_reader.label_map_path` | 标签映射文件路径                   | `label_map_path: "/home/user/workspace/label_map.pbtxt"`                                                                              |
| `train_input_reader.input_path`     | 训练集 TFRecord 路径            | `input_path: "/home/user/data/train.tfrecord"`                                                                                        |
| `eval_input_reader.label_map_path`  | 标签映射文件路径                   | `label_map_path: "/home/user/workspace/label_map.pbtxt"`                                                                              |
| `eval_input_reader.input_path`      | 验证集 TFRecord 路径            | `input_path: "/home/user/data/eval.tfrecord"`                                                                                         |
| `batch_size`                        | 批大小（CPU 建议 8，GPU 可 16\~32） | `batch_size: 8`                                                                                                                       |
| `num_steps`                         | 总训练步数                      | `num_steps: 50000`                                                                                                                    |
| `use_bfloat16`                      | CPU 训练必须设为 `false`         | `use_bfloat16: false`                                                                                                                 |

> **注意**：`eval_config` 中的 `num_examples` 可以不设置，默认使用全部验证集。

保存退出。

***

## 9. 启动训练

```bash
python $HOME/models/research/object_detection/model_main_tf2.py \
    --model_dir=$HOME/training \
    --pipeline_config_path=$HOME/workspace/pipeline.config \
    --num_train_steps=50000 \
    --alsologtostderr
```

训练开始后，每 100 步会输出一次损失值。日志示例：

```
Step 100 per-step time 0.561s
{'Loss/classification_loss': 0.2376, 'Loss/localization_loss': 0.2367, 'Loss/total_loss': 0.6277, ...}
```

***

## 10. 监控训练过程（TensorBoard）

打开另一个终端，激活环境并启动 TensorBoard：

```bash
source ~/SSD_cpu_gpu/bin/activate
tensorboard --logdir=~/training --port=6006
```

在 Windows 浏览器中访问 `http://localhost:6006` 即可看到损失曲线和评估指标（如果配置了评估）。

***

## 11. 暂停和恢复训练

- **暂停**：在训练终端按 `Ctrl+C`。脚本可能不会立即保存，但定期保存的 checkpoint 会保留。
- **恢复**：重新运行相同的训练命令，会自动从最新的 checkpoint 继续。

***

## 12. 评估模型（使用验证集/测试集）

### 12.1 使用训练好的 Checkpoint 评估

修改 `pipeline.config` 中 `eval_input_reader` 的 `input_path` 指向测试集文件（如 `test.tfrecord`），然后运行评估：

```bash
python $HOME/models/research/object_detection/model_main_tf2.py \
    --model_dir=$HOME/training \
    --pipeline_config_path=$HOME/workspace/pipeline.config \
    --checkpoint_dir=$HOME/training \
    --alsologtostderr
```

评估完成后，在 TensorBoard 的 “Scalars” 标签页会显示 `DetectionBoxes_Precision/mAP` 等指标。

### 12.2 使用 TFLite 模型评估（计算 mAP）

如果需要评估 INT8 量化后的 TFLite 模型，可以使用以下脚本 `evaluate_tflite.py`。该脚本会读取测试集 TFRecord，运行 TFLite 推理，并输出 COCO 风格的 mAP。

```python
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.metrics import coco_evaluation
import tqdm

MODEL_PATH = "/home/user/tflite/ssd_mobilenet_v2_fpnlite_int8.tflite"
TEST_TFRECORD = "/home/user/data/test.tfrecord"
LABEL_MAP_PATH = "/home/user/workspace/label_map.pbtxt"
NUM_TEST_IMAGES = -1
CONFIDENCE_THRESHOLD = 0.1

def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, (320, 320))
    image = tf.cast(image, tf.uint8)
    def to_dense(tensor):
        return tf.sparse.to_dense(tensor, default_value=0)
    xmin = to_dense(example['image/object/bbox/xmin'])
    ymin = to_dense(example['image/object/bbox/ymin'])
    xmax = to_dense(example['image/object/bbox/xmax'])
    ymax = to_dense(example['image/object/bbox/ymax'])
    classes = to_dense(example['image/object/class/label'])
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    return image, boxes, classes

def main():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 根据实际输出顺序设置索引（通常为：0: scores, 1: boxes, 2: num_detections, 3: classes）
    scores_idx = 0
    boxes_idx = 1
    num_detections_idx = 2
    classes_idx = 3

    label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=len(label_map.item), use_display_name=True)
    evaluator = coco_evaluation.CocoDetectionEvaluator(
        categories=categories,
        include_metrics_per_category=False   # 避免 per-category 缺失错误
    )

    dataset = tf.data.TFRecordDataset([TEST_TFRECORD])
    dataset = dataset.map(parse_tfrecord)
    if NUM_TEST_IMAGES > 0:
        dataset = dataset.take(NUM_TEST_IMAGES)

    for idx, (image, gt_boxes, gt_classes) in enumerate(tqdm.tqdm(dataset)):
        input_data = np.expand_dims(image.numpy(), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        num_detections = int(interpreter.get_tensor(output_details[num_detections_idx]['index']).flatten()[0])
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]

        num_detections = min(num_detections, len(scores))

        detection_boxes = []
        detection_scores = []
        detection_classes = []
        for i in range(num_detections):
            score = scores[i].item()
            if score > CONFIDENCE_THRESHOLD:
                cls_raw = classes[i].item()
                cls_id = int(cls_raw) + 1   # 0-based -> 1-based
                detection_boxes.append(boxes[i].tolist())
                detection_scores.append(float(score))
                detection_classes.append(cls_id)

        detection_boxes = np.array(detection_boxes, dtype=np.float32) if detection_boxes else np.empty((0, 4), dtype=np.float32)
        detection_scores = np.array(detection_scores, dtype=np.float32) if detection_scores else np.empty((0,), dtype=np.float32)
        detection_classes = np.array(detection_classes, dtype=np.int64) if detection_classes else np.empty((0,), dtype=np.int64)

        detected_dict = {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
        }

        gt_boxes_np = gt_boxes.numpy()
        gt_classes_np = gt_classes.numpy()
        groundtruth_boxes = []
        groundtruth_classes = []
        for i in range(len(gt_classes_np)):
            if gt_classes_np[i] == 0:
                continue
            groundtruth_boxes.append(gt_boxes_np[i].tolist())
            groundtruth_classes.append(int(gt_classes_np[i]))

        groundtruth_boxes = np.array(groundtruth_boxes, dtype=np.float32) if groundtruth_boxes else np.empty((0, 4), dtype=np.float32)
        groundtruth_classes = np.array(groundtruth_classes, dtype=np.int64) if groundtruth_classes else np.empty((0,), dtype=np.int64)

        groundtruth_dict = {
            'groundtruth_boxes': groundtruth_boxes,
            'groundtruth_classes': groundtruth_classes,
        }

        image_id = str(idx)
        evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dict)
        evaluator.add_single_detected_image_info(image_id, detected_dict)

    metrics = evaluator.evaluate()
    print("\n========== 评估结果 (test.tfrecord) ==========")
    for key, value in metrics.items():
        if 'Precision/mAP' in key or 'AR/' in key:
            print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()
```

运行评估：

```bash
source ~/SSD_cpu_gpu/bin/activate
cd ~/tflite
python evaluate_tflite.py
```

***

## 13. 导出浮点模型（SavedModel）

```bash
python $HOME/models/research/object_detection/exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path $HOME/workspace/pipeline.config \
    --trained_checkpoint_dir $HOME/training \
    --output_directory $HOME/exported_model
```

导出的 SavedModel 位于 `~/exported_model/saved_model`。

***

## 14. 模型量化为 INT8 并导出为 TFLite（基于实际验证的配置）

### 14.1 准备工作

- 确保已安装 `tensorflow-model-optimization`：
  ```bash
  source ~/SSD_cpu_gpu/bin/activate
  pip install tensorflow-model-optimization
  ```
- 确认训练已完成，`~/training` 目录下包含最新的 checkpoint 和 `pipeline.config`。
- 确认至少有一个 TFRecord 文件（训练集或验证集）可用于校准（通常 100\~500 张图片即可）。

### 14.2 一键转换脚本（最终实际可用版）

创建 `convert_to_int8.py` 文件：

```bash
mkdir -p ~/tflite
nano ~/tflite/convert_to_int8.py
```

将以下内容粘贴进去（注意修改开头的用户配置）：

```python
#!/usr/bin/env python3
"""
一键完成 SSD MobileNetV2 INT8 量化导出（最终实际可用版）
输入：uint8 [0,255]，输出：float32，顺序：scores, boxes, num_detections, classes
类别输出为 0-based float32，使用时需要 +1 转为 1-based
"""

import tensorflow as tf
import subprocess
import os
import sys
from pathlib import Path

# ==================== 用户配置区域 ====================
TRAINING_DIR = "/home/user/training"
PIPELINE_CONFIG_PATH = "/home/user/workspace/pipeline.config"
CALIB_TFRECORD = "/home/user/data/train.tfrecord"
NUM_CALIB_IMAGES = 300
INPUT_SIZE = (320, 320)
OUTPUT_DIR = "/home/user/tflite"
# ====================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
TFLITE_SAVED_MODEL = os.path.join(OUTPUT_DIR, "saved_model")
FINAL_TFLITE = os.path.join(OUTPUT_DIR, "ssd_mobilenet_v2_fpnlite_int8.tflite")

# 1. 导出 SavedModel
print("📦 步骤 1/2: 导出 TFLite 兼容的 SavedModel ...")
export_script = "/home/terry/models/research/object_detection/export_tflite_graph_tf2.py"
if not os.path.exists(export_script):
    print(f"❌ 错误: 找不到导出脚本 {export_script}")
    sys.exit(1)

cmd = [
    "python", export_script,
    "--pipeline_config_path", PIPELINE_CONFIG_PATH,
    "--trained_checkpoint_dir", TRAINING_DIR,
    "--output_directory", OUTPUT_DIR
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("❌ 导出 SavedModel 失败:", result.stderr)
    sys.exit(1)
print("✅ SavedModel 导出成功\n")

# 2. 量化转换
print("📦 步骤 2/2: INT8 量化转换 ...")

def parse_tfrecord(example_proto):
    # 校准只需要图像，不需要标签
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, INPUT_SIZE)
    # 模型输入要求 uint8 [0,255]，保持原始范围
    image = tf.cast(image, tf.uint8)
    return image

def representative_dataset():
    dataset = tf.data.TFRecordDataset([CALIB_TFRECORD])
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.take(NUM_CALIB_IMAGES)
    dataset = dataset.batch(1)
    for image in dataset:
        yield [image]   # image 是 uint8

converter = tf.lite.TFLiteConverter.from_saved_model(TFLITE_SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8          # 输入 uint8 [0,255]
converter.inference_output_type = tf.float32       # 输出 float32（scores, boxes, classes, num_detections）
converter.representative_dataset = representative_dataset

try:
    tflite_model = converter.convert()
    with open(FINAL_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"✅ INT8 模型已保存: {FINAL_TFLITE}")

    # 验证输出张量信息
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    print("\n📋 输出张量信息（实际顺序为：0=scores, 1=boxes, 2=num_detections, 3=classes）：")
    for i, out in enumerate(output_details):
        print(f"  输出 {i}: name={out['name']}, shape={out['shape']}, dtype={out['dtype']}")
except Exception as e:
    print(f"❌ 量化转换失败: {e}")
    sys.exit(1)

print("\n🎉 全部完成！")
```

### 14.3 检查 TFRecord 键名（重要）

脚本中的 `parse_tfrecord` 函数假设 TFRecord 中的图像键名为 `'image/encoded'`。如果你的 TFRecord 使用了其他键名（例如 `'image'` 或 `'img'`），需要修改。

可以使用以下代码查看你的 TFRecord 实际键名：

```bash
source ~/SSD_cpu_gpu/bin/activate
python -c "
import tensorflow as tf
tfrecord_path = '/home/user/data/train.tfrecord'
for raw_record in tf.data.TFRecordDataset([tfrecord_path]).take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(list(example.features.feature.keys()))
"
```

输出示例：`['image/encoded', 'image/height', 'image/width', ...]`

- 如果输出中包含 `'image/encoded'`，则无需修改。
- 如果输出中包含 `'image'` 或 `'img'`，则需要修改 `feature_description` 中的键名。

### 14.4 运行转换

```bash
cd ~/tflite
python convert_to_int8.py
```

转换成功后，最终模型保存在 `~/tflite/ssd_mobilenet_v2_fpnlite_int8.tflite`。

### 14.5 验证 INT8 模型（单张图片推理）

```python
import tensorflow as tf
import numpy as np
from PIL import Image

interpreter = tf.lite.Interpreter(model_path='~/tflite/ssd_mobilenet_v2_fpnlite_int8.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 根据实际输出顺序：0: scores, 1: boxes, 2: num_detections, 3: classes
scores_idx = 0
boxes_idx = 1
num_detections_idx = 2
classes_idx = 3

image = Image.open('test.jpg').resize((320, 320))
input_data = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
num_detections = int(interpreter.get_tensor(output_details[num_detections_idx]['index']).flatten()[0])
classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]

print(f"检测到 {num_detections} 个物体")
for i in range(min(num_detections, 5)):
    cls_raw = int(classes[i])          # 0,1,2,...
    cls_id = cls_raw + 1               # 转换为 1-based 标签
    print(f"  {i}: score={scores[i]:.3f}, class={cls_id}, box={boxes[i]}")
```

### 14.6 评估 INT8 模型（计算 mAP）

使用第 12.2 节提供的 `evaluate_tflite.py` 脚本，该脚本已经适配了输出顺序和类别转换。

***

## 15. 模型格式规范与使用指南（基于实际验证）

| 模型格式                  | 文件路径                                            | 输入要求                   | 输出说明                                                                                                 | 适用场景                   |
| :-------------------- | :---------------------------------------------- | :--------------------- | :--------------------------------------------------------------------------------------------------- | :--------------------- |
| **SavedModel（浮点）**    | `~/exported_model/saved_model`                  | 归一化像素值 `[0,1]`，float32 | 输出字典，包含 `detection_boxes` (float32), `detection_classes` (1-based int), `detection_scores` (float32) | 继续训练、评估、二次开发           |
| **TFLite INT8（混合量化）** | `~/tflite/ssd_mobilenet_v2_fpnlite_int8.tflite` | 像素值 `[0,255]`，uint8    | 输出顺序：`[scores, boxes, num_detections, classes]`，均为 float32；`classes` 为 **0-based** 浮点数（0.0,1.0,...）  | 边缘设备（如 limelight）低延迟部署 |

### 15.1 输出张量顺序（INT8 模型）

经过实际验证，导出的 INT8 TFLite 模型输出张量顺序为：

| 索引 | 名称                          | 形状           | 数据类型    | 含义                                  |
| :- | :-------------------------- | :----------- | :------ | :---------------------------------- |
| 0  | `StatefulPartitionedCall:1` | `[1, 10]`    | float32 | 检测置信度（0\~1）                         |
| 1  | `StatefulPartitionedCall:3` | `[1, 10, 4]` | float32 | 边界框 `[ymin, xmin, ymax, xmax]`（归一化） |
| 2  | `StatefulPartitionedCall:0` | `[1]`        | float32 | 有效检测框数量（通常固定为 10）                   |
| 3  | `StatefulPartitionedCall:2` | `[1, 10]`    | float32 | 类别索引（**0-based**，0.0,1.0,...）       |

### 15.2 类别索引转换

由于模型输出为 **0-based** 浮点数，而 `label_map.pbtxt` 中定义的是 **1-based** ID，使用时需要转换：

```python
cls_raw = int(classes[i])        # 0,1,2,...
cls_id = cls_raw + 1             # 转换为 1-based
```

### 15.3 部署到 limelight 等设备

- 确保设备端推理代码按照上述输出顺序解析张量。
- 输入图像需要预处理为 320×320 且像素值范围 `[0,255]`，类型 `uint8`。
- 输出置信度阈值建议设为 0.1\~0.3，根据实际需求调整。
- 若设备要求输出为 1-based 类别，可在后处理中加 1。

***

## 16. 评估脚本（evaluate\_tflite.py）

与第 12.2 节相同，这里再次确认其正确性：

```python
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.metrics import coco_evaluation
import tqdm

MODEL_PATH = "/home/user/tflite/ssd_mobilenet_v2_fpnlite_int8.tflite"
TEST_TFRECORD = "/home/user/data/test.tfrecord"
LABEL_MAP_PATH = "/home/user/workspace/label_map.pbtxt"
NUM_TEST_IMAGES = -1
CONFIDENCE_THRESHOLD = 0.1

def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, (320, 320))
    image = tf.cast(image, tf.uint8)
    def to_dense(tensor):
        return tf.sparse.to_dense(tensor, default_value=0)
    xmin = to_dense(example['image/object/bbox/xmin'])
    ymin = to_dense(example['image/object/bbox/ymin'])
    xmax = to_dense(example['image/object/bbox/xmax'])
    ymax = to_dense(example['image/object/bbox/ymax'])
    classes = to_dense(example['image/object/class/label'])
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    return image, boxes, classes

def main():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 输出顺序：0: scores, 1: boxes, 2: num_detections, 3: classes
    scores_idx = 0
    boxes_idx = 1
    num_detections_idx = 2
    classes_idx = 3

    label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=len(label_map.item), use_display_name=True)
    evaluator = coco_evaluation.CocoDetectionEvaluator(
        categories=categories,
        include_metrics_per_category=False
    )

    dataset = tf.data.TFRecordDataset([TEST_TFRECORD])
    dataset = dataset.map(parse_tfrecord)
    if NUM_TEST_IMAGES > 0:
        dataset = dataset.take(NUM_TEST_IMAGES)

    for idx, (image, gt_boxes, gt_classes) in enumerate(tqdm.tqdm(dataset)):
        input_data = np.expand_dims(image.numpy(), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        num_detections = int(interpreter.get_tensor(output_details[num_detections_idx]['index']).flatten()[0])
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]

        num_detections = min(num_detections, len(scores))

        detection_boxes = []
        detection_scores = []
        detection_classes = []
        for i in range(num_detections):
            score = scores[i].item()
            if score > CONFIDENCE_THRESHOLD:
                cls_raw = classes[i].item()
                cls_id = int(cls_raw) + 1   # 0-based -> 1-based
                detection_boxes.append(boxes[i].tolist())
                detection_scores.append(float(score))
                detection_classes.append(cls_id)

        detection_boxes = np.array(detection_boxes, dtype=np.float32) if detection_boxes else np.empty((0, 4), dtype=np.float32)
        detection_scores = np.array(detection_scores, dtype=np.float32) if detection_scores else np.empty((0,), dtype=np.float32)
        detection_classes = np.array(detection_classes, dtype=np.int64) if detection_classes else np.empty((0,), dtype=np.int64)

        detected_dict = {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
        }

        gt_boxes_np = gt_boxes.numpy()
        gt_classes_np = gt_classes.numpy()
        groundtruth_boxes = []
        groundtruth_classes = []
        for i in range(len(gt_classes_np)):
            if gt_classes_np[i] == 0:
                continue
            groundtruth_boxes.append(gt_boxes_np[i].tolist())
            groundtruth_classes.append(int(gt_classes_np[i]))

        groundtruth_boxes = np.array(groundtruth_boxes, dtype=np.float32) if groundtruth_boxes else np.empty((0, 4), dtype=np.float32)
        groundtruth_classes = np.array(groundtruth_classes, dtype=np.int64) if groundtruth_classes else np.empty((0,), dtype=np.int64)

        groundtruth_dict = {
            'groundtruth_boxes': groundtruth_boxes,
            'groundtruth_classes': groundtruth_classes,
        }

        image_id = str(idx)
        evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dict)
        evaluator.add_single_detected_image_info(image_id, detected_dict)

    metrics = evaluator.evaluate()
    print("\n========== 评估结果 (test.tfrecord) ==========")
    for key, value in metrics.items():
        if 'Precision/mAP' in key or 'AR/' in key:
            print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()
```

***

## 17. 常见错误及解决方法

| 错误                                                   | 原因             | 解决方法                                          |
| :--------------------------------------------------- | :------------- | :-------------------------------------------- |
| `ValueError: Category stats do not exist` (评估时)      | 测试集中某些类别缺失     | 设置 `include_metrics_per_category=False` 或忽略错误 |
| `IndexError: invalid index to scalar variable` (评估时) | 输出张量索引顺序错误     | 根据实际打印的 `output_details` 调整索引                 |
| 转换后模型输出类别全为 0 或 255                                  | 量化参数设置不当或类别未转换 | 使用本节提供的 `convert_to_int8.py`，并在评估时进行 `+1` 转换  |

***

## 18. 踩过的坑

- `port tensorflow as tf` 应当在 `from __future__` 之后。
- **INT8 模型输出顺序**为 `[scores, boxes, num_detections, classes]`，类别为 0-based float32。
- **评估脚本**使用正确的索引和类别转换，能够输出有效的 mAP。
- **GPU训练时PTX编译警告**：对于RTX 5070 Ti等新架构显卡，TensorFlow 2.15没有预编译内核，会触发`ptxas does not support CC 12.0`警告，并回退到驱动编译。这是预期行为，请耐心等待首次编译完成（约5-15分钟），后续训练速度正常。
- **环境变量LD\_LIBRARY\_PATH**：通过pip安装的CUDA库必须通过`LD_LIBRARY_PATH`暴露给TensorFlow，否则会报`libcudart.so`找不到的错误。务必按3.4.3节配置。
- **多问D老师：**完整开发日志<https://chat.deepseek.com/share/87xob6z57hebowsj0w>

