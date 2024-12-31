import json
import os
import numpy as np
from collections import defaultdict

# Load the cityscapes_panoptic_train.json file
with open('/mnt/source/cityscapes/annotations/cityscapes_panoptic_train_sam.json', 'r') as f:
    cityscapes_data = json.load(f)

# Define labels
labels = [
    ('unlabeled', 0), ('rectification border', 2), ('out of roi', 3), ('static', 4), 
    ('dynamic', 5), ('ground', 6), ('road', 7), ('sidewalk', 8), ('parking', 9), 
    ('rail track', 10), ('building', 11), ('wall', 12), ('fence', 13), 
    ('guard rail', 14), ('bridge', 15), ('tunnel', 16), ('pole', 17), 
    ('polegroup', 18), ('traffic light', 19), ('traffic sign', 20), 
    ('vegetation', 21), ('terrain', 22), ('sky', 23), ('person', 24), 
    ('rider', 25), ('car', 26), ('truck', 27), ('bus', 28), ('caravan', 29), 
    ('trailer', 30), ('train', 31), ('motorcycle', 32), ('bicycle', 33), 
    ('license plate', -1), ('lane', 34), ('left lane', 35), ('right lane', 36), 
    ('straight lane', 37), ('straight left lane', 38), ('straight right lane', 39), 
    ('right arrow', 40), ('left arrow', 41), ('straight arrow', 42), 
    ('straight left arrow', 43), ('straight right arrow', 44), ('ego vehicle', 1)
]

label_id_to_name = {label[1]: label[0] for label in labels}
label_name_to_id = {label[0]: label[1] for label in labels}

# Function to update mask labels
def update_mask_labels(label, label_list):
    matched_labels = [name for name, _ in label_list if name in label]
    return matched_labels if matched_labels else [label]

# IoU calculation function
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1, xi2, yi2 = max(x1, x1g), max(y1, y1g), min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# Create ground truth dictionary
gt_dict = defaultdict(list)
for annotation in cityscapes_data['annotations']:
    image_id = annotation['image_id']
    for segment in annotation['segments_info']:
        x, y, w, h = segment['bbox']
        gt_dict[image_id].append({
            'label': label_id_to_name[segment['category_id']],
            'bbox': [x, y, x + w, y + h]
        })

# Create prediction dictionary
pred_dict = defaultdict(list)
output_folder = '/mnt/source/Grounded-SAM-2/output_folder/'

for root, _, files in os.walk(output_folder):
    for file in files:
        if file == 'grounded_sam2_results.json':
            image_id = os.path.basename(root).replace('_leftImg8bit', '')
            #print(image_id)
            with open(os.path.join(root, file), 'r') as f:
                mask_data = json.load(f)
                for entry in mask_data['annotations']:
                    #print(entry, "entry")
                    if 'bbox' in entry:
                        updated_labels = update_mask_labels(entry['class_name'], labels)
                        for label in updated_labels:
                            pred_dict[image_id].append({
                                'label': label,
                                'box': entry['bbox']
                            })

# Calculate mAP and mIoU
def calculate_metrics_per_class(gt_dict, pred_dict, iou_threshold=0.8):
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0})
    for image_id in gt_dict:
        gt_labels = gt_dict[image_id]
        pred_labels = pred_dict.get(image_id, [])
        matched_pred_indices = set()
        matched_gt_indices = set()

        for i, gt in enumerate(gt_labels):
            gt_label = gt['label']
            gt_bbox = gt['bbox']
            found_match = False
            max_iou = 0.0
            max_iou_pred_idx = -1

            for j, pred in enumerate(pred_labels):
                if j in matched_pred_indices:
                    continue  # 既にマッチした予測はスキップ

                pred_label = pred['label']
                pred_box = pred['box']

                if gt_label == pred_label:
                    iou = calculate_iou(gt_bbox, pred_box)
                    if iou >= iou_threshold and iou > max_iou:
                        max_iou = iou
                        max_iou_pred_idx = j
                        found_match = True

            if found_match:
                class_metrics[gt_label]['tp'] += 1
                class_metrics[gt_label]['iou_sum'] += max_iou
                matched_pred_indices.add(max_iou_pred_idx)
                matched_gt_indices.add(i)
            else:
                class_metrics[gt_label]['fn'] += 1

        # マッチしなかった予測をFPとしてカウント
        for j, pred in enumerate(pred_labels):
            if j not in matched_pred_indices:
                pred_label = pred['label']
                class_metrics[pred_label]['fp'] += 1

    ap_per_class = {}
    miou_per_class = {}
    for label, metrics in class_metrics.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        iou_sum = metrics['iou_sum']
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        # mIoUの計算
        miou = iou_sum / tp if tp > 0 else 0.0

        ap_per_class[label] = precision  # 正確なAPを計算する場合は、別途計算が必要
        miou_per_class[label] = miou

    return ap_per_class, miou_per_class


def calculate_map_at_60(gt_dict, pred_dict):
    return calculate_metrics_per_class(gt_dict, pred_dict, iou_threshold=0.8)

map_per_class, iou_per_class = calculate_metrics_per_class(gt_dict, pred_dict)
map_at_60_per_class, _ = calculate_map_at_60(gt_dict, pred_dict)

for label in sorted(map_per_class.keys(), key=lambda x: label_name_to_id.get(x, float('inf'))):
    if label in label_name_to_id:
        print(f'Class: {label}')
        print(f'  mAP: {map_per_class[label]:.4f}')
        print(f'  mIoU: {iou_per_class[label]:.4f}')
        print(f'  mAP@60: {map_at_60_per_class[label]:.4f}')
