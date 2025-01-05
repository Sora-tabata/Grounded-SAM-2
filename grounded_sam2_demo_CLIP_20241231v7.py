import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import clip
from PIL import Image
import torch.nn.functional as F

def iou(bbox1, bbox2):
    """
    bbox: (x_min, y_min, x_max, y_max) の形式を想定
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # 交差領域 (Intersection)
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    intersection = inter_w * inter_h
    
    # bbox1 と bbox2 の面積 (Union の計算に使う)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # IoU = Intersection / Union
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union

class CLIPRelabeler:
    def __init__(self, clip_checkpoint, device='cuda'):
        """CLIPモデルを使用して画像領域を分類するクラス"""
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.model = self.model.float()
        state_dict = torch.load(clip_checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.class_names, self.class_stats = self._load_class_info()

    def _load_class_info(self):
        """クラス情報とその統計を読み込む"""
        labels_path = '/mnt/source/Downloads/labels_total.json'
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)

        class_counts = {}
        for entry in labels_data:
            label = entry['label']
            class_counts[label] = class_counts.get(label, 0) + 1

        unique_classes = list(class_counts.keys())
        total_samples = sum(class_counts.values())
        
        class_stats = {
            'counts': class_counts,
            'total_samples': total_samples,
            'avg_length': sum(len(name.split()) for name in unique_classes) / len(unique_classes)
        }

        return unique_classes, class_stats

    def segment_image(self, image, mask):
        mask = mask.astype(bool)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        image_array = np.array(image)
        segmented = np.zeros_like(image_array)
        segmented[mask] = image_array[mask]
        
        black_image = Image.new("RGB", image.size, (0, 0, 0))
        transparency = np.zeros_like(mask, dtype=np.uint8)
        transparency[mask] = 255
        transparency = Image.fromarray(transparency, mode='L')
        
        segmented = Image.fromarray(segmented)
        black_image.paste(segmented, mask=transparency)
        return black_image


    def _normalize_similarities(self, similarity_matrix):
        """
        類似度スコアを正規化する関数です。
        
        以下の要因を考慮して正規化を行います：
        1. クラス名の長さによるバイアス
        2. クラスの出現頻度
        3. 特徴空間での分布
        """
        # 分布の正規化
        similarity_std = similarity_matrix.std(dim=1, keepdim=True)
        similarity_mean = similarity_matrix.mean(dim=1, keepdim=True)
        normalized_similarity = (similarity_matrix - similarity_mean) / (similarity_std + 1e-6)
        
        # クラス特性による重み付け
        weights = torch.ones_like(normalized_similarity[0])
        for i, class_name in enumerate(self.class_names):
            # クラス名の長さによる調整
            length_factor = 1.0 / (1.0 + len(class_name.split()))
            
            # 出現頻度による調整
            freq = self.class_stats['counts'][class_name]
            freq_factor = np.log1p(freq) / np.log1p(self.class_stats['total_samples'])
            
            # 重みの組み合わせ
            weights[i] = length_factor * (1.0 + 0.5 * freq_factor)
        
        # 最終的な正規化スコアの計算
        weighted_similarity = normalized_similarity * weights.to(self.device)
        
        # スケーリング
        temperature = 0.07
        return weighted_similarity / temperature

    @torch.no_grad()
    def relabel_regions(self, image, masks, confidence_threshold=0.02):
        """
        画像領域に新しいラベルを付与する関数です。
        
        各領域に対して:
        1. CLIPモデルを使用して特徴を抽出
        2. 正規化された類似度スコアを計算
        3. 最も類似度の高いクラスを選択
        """
        # マスク領域の抽出と前処理
        masked_regions = [self.segment_image(image, mask) for mask in masks]
        processed_images = torch.stack([
            self.preprocess(img) for img in masked_regions
        ]).to(self.device)

        with torch.cuda.amp.autocast():
            # 画像とテキストの特徴抽出
            image_features = self.model.encode_image(processed_images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            text_tokens = clip.tokenize(self.class_names).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # 類似度の計算と調整
            similarity = (100.0 * image_features @ text_features.T)
            adjusted_similarity = self._normalize_similarities(similarity)
            probs = F.softmax(adjusted_similarity, dim=-1)

            # バッチ処理での予測結果の取得
            predictions = []
            for i in range(len(masks)):
                region_probs = probs[i].cpu().numpy()
                
                # 最も高いスコアを持つクラスを選択
                selected_idx = np.argmax(region_probs)
                selected_class = self.class_names[selected_idx]
                confidence = float(region_probs[selected_idx])
                
                # 上位3クラスも記録したい場合はここで取得
                top_k = min(3, len(self.class_names))
                top_indices = np.argsort(region_probs)[-top_k:][::-1]
                top_probs = region_probs[top_indices]
                
                # しきい値より低い場合は unknown とする
                if confidence < confidence_threshold:
                    selected_class = "unknown"
                
                # 予測結果の記録
                predictions.append({
                    'class_name': selected_class,
                    'confidence': confidence,
                    'top_predictions': [
                        {
                            'class': self.class_names[idx],
                            'probability': float(prob),
                            'adjusted_score': float(prob * (1.0 / (1.0 + len(self.class_names[idx].split()))))
                        }
                        for idx, prob in zip(top_indices, top_probs)
                    ]
                })

        return predictions, masked_regions


class ResultVisualizer:
    def __init__(self):
        """検出結果の可視化を行うクラスです。
        このクラスは、物体検出、セグメンテーション、およびラベリングの結果を
        視覚的に表現するための機能を提供します。
        """
        pass

    def visualize(self, image, boxes, masks, labels):
        """検出結果を可視化します。
        
        Args:
            image: 入力画像（RGB形式）
            boxes: バウンディングボックスの座標
            masks: セグメンテーションマスク
            labels: 予測されたクラスラベルとスコア
        
        Returns:
            annotated_frame: アノテーション付きの画像
        """
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks.astype(bool),
            class_id=np.arange(len(boxes))  # 連番でクラスIDを割り当て
        )

        annotated_frame = image.copy()
        
        # マスクの描画（透明度30%）
        mask_annotator = sv.MaskAnnotator(opacity=0.3)
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        
        # バウンディングボックスの描画（線の太さ2ピクセル）
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        
        # ラベルの描画（テキストの太さと大きさを調整）
        label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=0.6
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        return annotated_frame


def mask2rle(mask):
    """セグメンテーションマスクをRLE (Run-Length Encoding) 形式に変換します。"""
    rle = mask_util.encode(
        np.array(mask[:, :, None], order="F", dtype="uint8")
    )[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_image(args):
    """画像処理のメインパイプラインを実行します。"""
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # モデルの初期化
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=device
    )
    
    clip_relabeler = CLIPRelabeler(args.clip_checkpoint, device)
    visualizer = ResultVisualizer()

    # 画像の読み込みと前処理
    image_source, image = load_image(args.input_image)
    sam2_predictor.set_image(image_source)

    # Grounding DINOでの物体検出
    boxes, _, _ = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device
    )

    # ボックスの座標変換
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # SAM2でのセグメンテーション
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # CLIPによる領域の再ラベリング（全マスクに対して推論し、最大スコアのクラスを付与）
    predictions, masked_regions = clip_relabeler.relabel_regions(
        image_source, 
        masks,
        confidence_threshold=0.02
    )

    # ラベルの生成
    labels = [
        f"{pred['class_name']} ({pred['confidence']:.2f})"
        for pred in predictions
    ]

    # 結果の可視化
    annotated_frame = visualizer.visualize(
        image_source,
        input_boxes,
        masks,
        labels
    )

    # 可視化結果の保存
    output_path = os.path.join(args.output_dir, "relabeled_result.jpg")
    cv2.imwrite(
        output_path,
        cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    )

    # クロップ画像の保存
    crops_dir = os.path.join(args.output_dir, 'cropped_regions')
    os.makedirs(crops_dir, exist_ok=True)
    for i, (region, pred) in enumerate(zip(masked_regions, predictions)):
        region.save(os.path.join(
            crops_dir,
            f"region_{i:03d}_{pred['class_name']}_{pred['confidence']:.2f}.png"
        ))

    # JSON結果の保存（オプション）
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "image_size": {"width": w, "height": h},
            "predictions": [
                {
                    "region_id": i,
                    "class_name": pred["class_name"],
                    "confidence": pred["confidence"],
                    "bbox": box.tolist(),
                    "mask_rle": mask2rle(mask)
                }
                for i, (pred, box, mask) in enumerate(zip(predictions, input_boxes, masks))
            ]
        }
        
        with open(os.path.join(args.output_dir, "detection_results.json"), "w") as f:
            json.dump(results, f, indent=4)

    print(f"Processing completed! Results saved to {args.output_dir}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser("Grounded SAM2 with Fine-tuned CLIP Demo", add_help=True)
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_config", type=str, required=True, help="Path to SAM2 config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="Path to Grounding DINO checkpoint")
    parser.add_argument("--grounded_config", type=str, required=True, help="Path to Grounding DINO config file")
    parser.add_argument("--clip_checkpoint", type=str, required=True, help="Path to fine-tuned CLIP checkpoint")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for Grounding DINO")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--dump_json", action="store_true", help="Dump results to JSON file")
    
    args = parser.parse_args()
    process_image(args)


if __name__ == "__main__":
    main()
