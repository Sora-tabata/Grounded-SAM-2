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

# --- 追加: Tip-Adapter ロードのため ---
import clip
import torch.nn.functional as F
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# ここで使用するクラスラベルを定義
selected_labels = [
    "Speed Limit 30 km/h", 
    "Main road",
    "Yield",
    "Stop",
    "No vehicles",
    "No entry",
    "Pedestrians",
    "Turn right",
    "Turn left",
    "Only straight",
    "Keep right",
    "Keep left",
    "Road straight arrow marker",
    "Road right arrow marker",
    "Road straight left arrow marker",
    "Road left arrow marker"    
]
# =================== ここから Tip-Adapter を用いたモデル作成部分 ===================

class TipAdapterCLIP(torch.nn.Module):
    """
    Tip-Adapter の重みが読み込まれた CLIP + Adapter モデルを統合したクラス。
    - encode_image / encode_text メソッドで特徴を取得できるようにする
    """
    def __init__(self, base_clip_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = device
        # ベースCLIPをロード
        self.clip_model, self.preprocess = clip.load(base_clip_name, device=self.device, jit=False)
        self.clip_model = self.clip_model.float().eval()
        
        # Adapter の部分を後で読み込む想定
        self.adapter = torch.nn.ModuleDict()
    
    def load_tip_adapter(self, tip_adapter_checkpoint):
        """
        Tip-Adapter のチェックポイントを読み込む。
        通常、CLIP + Adapter の両方が含まれている場合が多い。
        """
        state_dict = torch.load(tip_adapter_checkpoint, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded Tip-Adapter checkpoint from {tip_adapter_checkpoint}")
    
    @torch.no_grad()
    def encode_image(self, images):
        """Tip-Adapter 付きの encode_image"""
        base_feat = self.clip_model.encode_image(images)
        return base_feat

    @torch.no_grad()
    def encode_text(self, texts):
        """Tip-Adapter 付きの encode_text"""
        base_feat = self.clip_model.encode_text(texts)
        return base_feat


class CLIPRelabeler:
    def __init__(self, tip_adapter_checkpoint, device='cuda'):
        """CLIPモデルを使用して画像領域を分類するクラス"""
        self.device = device
        
        self.tip_model = TipAdapterCLIP("ViT-B/32", device=device)
        self.tip_model.load_tip_adapter(tip_adapter_checkpoint)
        self.tip_model.eval()
        
        self.preprocess = self.tip_model.preprocess
        
        self.class_names, self.class_stats = self._load_class_info()

    def _load_class_info(self):
        """クラス情報とその統計を読み込む (selected_labels のみ)"""
        labels_path = '/mnt/source/Downloads/labels_total.json'
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)

        class_counts = {}
        for entry in labels_data:
            label = entry['label']
            # selected_labels 以外はスキップ
            if label not in selected_labels:
                continue
            class_counts[label] = class_counts.get(label, 0) + 1

        unique_classes = list(class_counts.keys())
        total_samples = sum(class_counts.values())

        # 統計情報が不要なら削ってもOK
        class_stats = {
            'counts': class_counts,
            'total_samples': total_samples,
            'avg_length': sum(len(name.split()) for name in unique_classes) / len(unique_classes) if unique_classes else 0
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
        similarity_std = similarity_matrix.std(dim=1, keepdim=True)
        similarity_mean = similarity_matrix.mean(dim=1, keepdim=True)
        normalized_similarity = (similarity_matrix - similarity_mean) / (similarity_std + 1e-6)
        
        weights = torch.ones_like(normalized_similarity[0])
        for i, class_name in enumerate(self.class_names):
            length_factor = 1.0 / (1.0 + len(class_name.split()))
            freq = self.class_stats['counts'].get(class_name, 0)
            freq_factor = np.log1p(freq) / np.log1p(self.class_stats['total_samples']+1e-6) if self.class_stats['total_samples']>0 else 0
            weights[i] = length_factor * (1.0 + 0.5 * freq_factor)
        
        weighted_similarity = normalized_similarity * weights.to(self.device)
        
        temperature = 0.07
        return weighted_similarity / temperature

    @torch.no_grad()
    def relabel_regions(self, image, masks, confidence_threshold=0.02):
        masked_regions = [self.segment_image(image, mask) for mask in masks]
        processed_images = torch.stack([
            self.preprocess(img) for img in masked_regions
        ]).to(self.device)

        with torch.cuda.amp.autocast():
            image_features = self.tip_model.encode_image(processed_images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # self.class_names が selected_labels に絞られている
            text_tokens = clip.tokenize(self.class_names).to(self.device)
            text_features = self.tip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T)
            adjusted_similarity = self._normalize_similarities(similarity)
            probs = F.softmax(adjusted_similarity, dim=-1)

            predictions = []
            for i in range(len(masks)):
                region_probs = probs[i].cpu().numpy()
                selected_idx = np.argmax(region_probs)
                selected_class = self.class_names[selected_idx]
                confidence = float(region_probs[selected_idx])
                
                top_k = min(3, len(self.class_names))
                top_indices = np.argsort(region_probs)[-top_k:][::-1]
                top_probs = region_probs[top_indices]
                
                if confidence < confidence_threshold:
                    selected_class = "unknown"
                
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
        pass

    def visualize(self, image, boxes, masks, labels):
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks.astype(bool),
            class_id=np.arange(len(boxes))
        )

        annotated_frame = image.copy()
        
        mask_annotator = sv.MaskAnnotator(opacity=0.3)
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        
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
    rle = mask_util.encode(
        np.array(mask[:, :, None], order="F", dtype="uint8")
    )[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_image(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=device
    )
    
    clip_relabeler = CLIPRelabeler(args.clip_checkpoint, device)
    visualizer = ResultVisualizer()

    image_source, image = load_image(args.input_image)
    sam2_predictor.set_image(image_source)

    boxes, _, _ = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

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

    predictions, masked_regions = clip_relabeler.relabel_regions(
        image_source, 
        masks,
        confidence_threshold=0.02
    )

    labels = [
        f"{pred['class_name']} ({pred['confidence']:.2f})"
        for pred in predictions
    ]

    annotated_frame = visualizer.visualize(
        image_source,
        input_boxes,
        masks,
        labels
    )

    output_path = os.path.join(args.output_dir, "relabeled_result.jpg")
    cv2.imwrite(
        output_path,
        cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    )

    crops_dir = os.path.join(args.output_dir, 'cropped_regions')
    os.makedirs(crops_dir, exist_ok=True)
    for i, (region, pred) in enumerate(zip(masked_regions, predictions)):
        region.save(os.path.join(
            crops_dir,
            f"region_{i:03d}_{pred['class_name']}_{pred['confidence']:.2f}.png"
        ))

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
    parser = argparse.ArgumentParser("Grounded SAM2 with Tip-Adapter Demo", add_help=True)
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_config", type=str, required=True, help="Path to SAM2 config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="Path to Grounding DINO checkpoint")
    parser.add_argument("--grounded_config", type=str, required=True, help="Path to Grounding DINO config file")
    parser.add_argument("--clip_checkpoint", type=str, required=True, help="Path to Tip-Adapter checkpoint")
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
