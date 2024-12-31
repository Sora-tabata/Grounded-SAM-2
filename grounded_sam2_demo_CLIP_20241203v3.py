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

# メモリの解放
torch.cuda.empty_cache()

def single_mask_to_rle(mask):
    """マスクをRLE形式に変換"""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def segment_image(image, segmentation_mask):
    """
    セグメンテーションマスクを使用して画像から対象領域を抽出し、
    黒背景上に配置します。
    """
    segmentation_mask = segmentation_mask.astype(bool)
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

class CLIPClassifier:
    def __init__(self, model_path, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.model = self.model.float()
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # クラス名とその統計情報を読み込み
        self.class_names, self.class_stats = self._load_class_info()

    def _load_class_info(self):
        """
        クラス名と、各クラスの統計情報を読み込みます。
        出現頻度や平均的な単語数なども計算します。
        """
        labels_path = '/mnt/source/Downloads/labels_total.json'
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)

        # クラスの出現回数をカウント
        class_counts = {}
        for entry in labels_data:
            label = entry['label']
            class_counts[label] = class_counts.get(label, 0) + 1

        # ユニークなクラス名のリストを作成
        unique_classes = list(class_counts.keys())
        
        # クラスごとの統計情報を計算
        total_samples = sum(class_counts.values())
        class_stats = {
            'counts': class_counts,
            'total_samples': total_samples,
            'avg_length': sum(len(name.split()) for name in unique_classes) / len(unique_classes)
        }

        return unique_classes, class_stats

    def _normalize_similarities(self, similarity_matrix):
        """
        類似度スコアを穏やかに調整します。
        極端な調整を避け、バランスの取れたスコアを返します。
        """
        # クラス名の長さに基づく軽い調整
        lengths = torch.tensor([
            min(len(name.split()), 4) / 4  # 長さの影響を制限
            for name in self.class_names
        ], device=self.device)
        
        # より穏やかな調整係数を適用
        length_factors = 1.0 - (0.1 * lengths)  # 最大で10%のペナルティ
        
        # クラスの出現頻度による軽い調整
        freq_factors = torch.tensor([
            (self.class_stats['total_samples'] / self.class_stats['counts'][name]) ** 0.1  # べき乗で影響を抑制
            for name in self.class_names
        ], device=self.device)
        
        # 調整係数を正規化
        freq_factors = freq_factors / freq_factors.max()
        
        # 両方の調整を組み合わせて適用
        combined_factors = length_factors * freq_factors
        adjusted_similarity = similarity_matrix * combined_factors.unsqueeze(0)
        
        return adjusted_similarity

    @torch.no_grad()
    def classify_regions(self, masked_images, confidence_threshold=0.02):
        """
        マスクされた領域を分類します。
        より穏やかな閾値と調整を適用し、極端な結果を避けます。
        """
        processed_images = torch.stack([
            self.preprocess(img) for img in masked_images
        ]).to(self.device)

        with torch.cuda.amp.autocast():
            # 特徴量の抽出と正規化
            image_features = self.model.encode_image(processed_images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            text_tokens = clip.tokenize(self.class_names).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # 基本の類似度計算
            similarity = (100.0 * image_features @ text_features.T)
            
            # 穏やかな調整を適用
            adjusted_similarity = self._normalize_similarities(similarity)
            
            # 温度パラメータを使用して確率に変換
            temperature = 0.07
            probs = F.softmax(adjusted_similarity * temperature, dim=-1)

            # 最も確信度の高いクラスとスコアを取得
            confidence_scores, indices = probs.max(dim=1)

        predicted_classes = []
        final_scores = []
        
        # より低い閾値を使用して結果を判定
        for score, idx in zip(confidence_scores.cpu().numpy(), indices.cpu().numpy()):
            if score >= confidence_threshold:
                predicted_classes.append(self.class_names[idx])
                final_scores.append(score)
            else:
                predicted_classes.append("unknown")
                final_scores.append(score)

        return predicted_classes, np.array(final_scores)
    
def main():
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

    # 設定と初期化
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # モデルの構築
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=device
    )

    # CLIPクラス分類器の初期化
    clip_classifier = CLIPClassifier(args.clip_checkpoint, device)

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
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # セグメント領域の抽出
    pil_image = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
    masked_regions = [
        segment_image(pil_image, mask)
        for mask in masks
    ]

    # CLIPによる領域の分類
    predicted_classes, confidence_scores = clip_classifier.classify_regions(masked_regions)

    # 検出結果のラベル作成
    labels = [
        f"{class_name} {score:.2f}"
        for class_name, score in zip(predicted_classes, confidence_scores)
    ]

    # 結果の可視化
    img = cv2.imread(args.input_image)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=np.arange(len(confidence_scores))
    )

    # アノテーションの作成
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 可視化結果の保存
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    cv2.imwrite(
        os.path.join(args.output_dir, "groundingdino_annotated_image.jpg"),
        annotated_frame
    )

    annotated_frame_with_mask = mask_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    cv2.imwrite(
        os.path.join(args.output_dir, "grounded_sam2_annotated_image_with_mask.jpg"),
        annotated_frame_with_mask
    )

    # JSON形式での結果保存
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box.tolist(),
                    "segmentation": single_mask_to_rle(mask),
                    "score": float(score),
                }
                for class_name, box, mask, score in zip(
                    predicted_classes, input_boxes, masks, confidence_scores
                )
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

        with open(os.path.join(args.output_dir, "grounded_sam2_results.json"), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()