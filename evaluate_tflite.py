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

    # 根据打印信息手动设置索引
    # 0: scores, 1: boxes, 2: num_detections, 3: classes
    scores_idx = 0
    boxes_idx = 1
    num_detections_idx = 2
    classes_idx = 3

    label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=len(label_map.item), use_display_name=True)
    evaluator = coco_evaluation.CocoDetectionEvaluator(
        categories=categories,
        include_metrics_per_category=True
    )

    dataset = tf.data.TFRecordDataset([TEST_TFRECORD])
    dataset = dataset.map(parse_tfrecord)
    if NUM_TEST_IMAGES > 0:
        dataset = dataset.take(NUM_TEST_IMAGES)

    for idx, (image, gt_boxes, gt_classes) in enumerate(tqdm.tqdm(dataset)):
        input_data = np.expand_dims(image.numpy(), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 获取输出
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]          # (10,)
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]            # (10,4)
        num_detections = int(interpreter.get_tensor(output_details[num_detections_idx]['index']).flatten()[0])
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]        # (10,)

        # 调试：打印前几张图片的信息
        if idx < 5:
            print(f"\nImage {idx}: num_detections={num_detections}")
            print(f"  scores[:5]: {scores[:5]}")
            print(f"  classes[:5]: {classes[:5]}")
            print(f"  boxes[:5]: {boxes[:5]}")

        max_detections = len(scores)
        num_detections = min(num_detections, max_detections)

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

        # 真实标注
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