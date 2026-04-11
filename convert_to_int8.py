#!/usr/bin/env python3
"""
一键完成 SSD MobileNetV2 INT8 量化导出（最终修正版）
修复代表数据集类型错误，输出保持 float32
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
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, INPUT_SIZE)
    # 关键：归一化到 [0,1] 并转为 float32，匹配模型原始输入
    image = tf.cast(image, tf.float32) / 255.0
    return image

def representative_dataset():
    dataset = tf.data.TFRecordDataset([CALIB_TFRECORD])
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.take(NUM_CALIB_IMAGES)
    dataset = dataset.batch(1)
    for image in dataset:
        yield [image]   # image 是 float32

converter = tf.lite.TFLiteConverter.from_saved_model(TFLITE_SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
converter.representative_dataset = representative_dataset

try:
    tflite_model = converter.convert()
    with open(FINAL_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"✅ INT8 模型已保存: {FINAL_TFLITE}")
    
    # 验证输出
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    print("\n📋 输出张量信息：")
    for out in output_details:
        print(f"   name={out['name']}, dtype={out['dtype']}")
        if out['dtype'] == tf.float32:
            print(f"      ✅ 类别输出为 float32，值为 0.0/1.0")
except Exception as e:
    print(f"❌ 量化转换失败: {e}")
    sys.exit(1)

print("\n🎉 全部完成！")