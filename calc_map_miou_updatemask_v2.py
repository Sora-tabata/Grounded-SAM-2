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
            with open(os.path.join(root, file), 'r') as f:
                mask_data = json.load(f)
                for entry in mask_data['annotations']:
                    if 'bbox' in entry:
                        updated_labels = update_mask_labels(entry['class_name'], labels)
                        for label in updated_labels:
                            pred_dict[image_id].append({
                                'label': label,
                                'box': entry['bbox'],
                                'score': entry['score'][0]  # 信頼度スコアを追加
                            })

def calculate_ap(recalls, precisions):
    # Append sentinel values at the end
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # Integrate area under curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

# Calculate mAP and mIoU
def calculate_metrics_per_class(gt_dict, pred_dict, iou_threshold=0.5):
    gt_classes = defaultdict(list)
    pred_classes = defaultdict(list)

    # Collect ground truths per class
    for image_id in gt_dict:
        for gt in gt_dict[image_id]:
            gt_label = gt['label']
            gt_classes[gt_label].append({
                'image_id': image_id,
                'bbox': gt['bbox'],
                'used': False  # To keep track of matched ground truths
            })

    # Collect predictions per class
    for image_id in pred_dict:
        for pred in pred_dict[image_id]:
            pred_label = pred['label']
            pred_classes[pred_label].append({
                'image_id': image_id,
                'bbox': pred['box'],
                'score': pred['score']
            })

    metrics_per_class = {}
    for label in gt_classes.keys() | pred_classes.keys():
        gt_bboxes = gt_classes.get(label, [])
        pred_bboxes = pred_classes.get(label, [])
        npos = len(gt_bboxes)
        # Sort predictions by confidence score in descending order
        pred_bboxes = sorted(pred_bboxes, key=lambda x: -x['score'])
        tp = np.zeros(len(pred_bboxes))
        fp = np.zeros(len(pred_bboxes))
        gt_used = {}
        for idx_pred, pred in enumerate(pred_bboxes):
            image_id = pred['image_id']
            pred_box = pred['bbox']
            max_iou = 0
            max_gt_idx = -1
            for idx_gt, gt in enumerate(gt_bboxes):
                if gt['image_id'] != image_id:
                    continue
                if gt_used.get(idx_gt, False):
                    continue
                iou = calculate_iou(pred_box, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = idx_gt
            if max_iou >= iou_threshold:
                tp[idx_pred] = 1
                gt_used[max_gt_idx] = True
            else:
                fp[idx_pred] = 1
        # Compute cumulative TP and FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / npos if npos > 0 else np.zeros(len(cum_tp))
        precisions = cum_tp / (cum_tp + cum_fp + 1e-16)
        ap = calculate_ap(recalls, precisions)
        metrics_per_class[label] = {
            'ap': ap,
            'num_gt': npos,
            'num_pred': len(pred_bboxes),
            'recalls': recalls,
            'precisions': precisions,
            'tp': int(sum(tp)),
            'fp': int(sum(fp)),
            'fn': npos - int(sum(tp)),
            'precision': precisions[-1] if len(precisions) > 0 else 0.0,
            'recall': recalls[-1] if len(recalls) > 0 else 0.0
        }
    return metrics_per_class

# IoU thresholds for evaluation
iou_thresholds = [0.5, 0.6]
metrics_per_class_all = {}

for iou_threshold in iou_thresholds:
    metrics_per_class = calculate_metrics_per_class(gt_dict, pred_dict, iou_threshold=iou_threshold)
    metrics_per_class_all[iou_threshold] = metrics_per_class

# Print results
for label in sorted(label_name_to_id.keys(), key=lambda x: label_name_to_id.get(x, float('inf'))):
    if label in label_name_to_id:
        print(f'Class: {label}')
        for iou_threshold in iou_thresholds:
            metrics = metrics_per_class_all[iou_threshold].get(label, None)
            if metrics:
                print(f'  IoU Threshold: {iou_threshold}')
                print(f'    Number of GT instances: {metrics["num_gt"]}')
                print(f'    Number of predicted instances: {metrics["num_pred"]}')
                print(f'    True Positives: {metrics["tp"]}')
                print(f'    False Positives: {metrics["fp"]}')
                print(f'    False Negatives: {metrics["fn"]}')
                print(f'    Precision: {metrics["precision"]:.4f}')
                print(f'    Recall: {metrics["recall"]:.4f}')
                print(f'    AP: {metrics["ap"]:.4f}')
            else:
                print(f'  IoU Threshold: {iou_threshold}')
                print(f'    No data available for this class.')