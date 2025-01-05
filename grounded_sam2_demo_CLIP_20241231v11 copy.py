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

import clip
import torch.nn.functional as F
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# ================== クラスラベル ==================
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

class TipAdapterRelabeler:
    """
    Tip-Adapter 学習済みキャッシュをロードし、SAM2 で切り出した画像を CLIP エンコード + cache で推論。
    """
    def __init__(self, tip_adapter_checkpoint, device='cuda'):
        self.device = device
        # ベースCLIP読み込み
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.clip_model.eval().float()

        # 学習済みキャッシュを読み込み
        ckpt = torch.load(tip_adapter_checkpoint, map_location=self.device)
        print(f"[INFO] Loaded Tip-Adapter cache from: {tip_adapter_checkpoint}")

        # ここで .float() を付与して、すべて float32 に統一
        self.cache_keys   = ckpt['cache_keys'].to(self.device).float()     # (dim, N)
        self.cache_values = ckpt['cache_values'].to(self.device).float()   # (N, C)
        self.clip_weights = ckpt['clip_weights'].to(self.device).float()   # (dim, num_classes)

        self.best_alpha = ckpt['best_alpha']
        self.best_beta  = ckpt['best_beta']

        if 'selected_labels' in ckpt:
            self.class_names = ckpt['selected_labels']
        else:
            self.class_names = selected_labels

    @torch.no_grad()
    def _encode_image(self, pil_image: Image.Image):
        """
        PIL.Image -> CLIP 画像特徴 (float32, L2正規化)
        """
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        # encode_image の結果が half になる場合があるが、ここで明示的に float() に変換
        feat = self.clip_model.encode_image(img_tensor).float()
        feat = F.normalize(feat, p=2, dim=-1)
        return feat  # (1, dim)

    @torch.no_grad()
    def _tip_adapter_inference(self, image_features: torch.Tensor):
        """
        Tip-Adapter ロジック:
          clip_logits = 100. * (image_features @ clip_weights)
          affinity = image_features @ cache_keys
          cache_logits = exp(-1 * (beta - beta*affinity)) @ cache_values
          final_logits = clip_logits + alpha * cache_logits
        """
        # image_features, clip_weights も float32 同士
        clip_logits = 100.0 * (image_features @ self.clip_weights)  # (1, num_classes)

        affinity = image_features @ self.cache_keys  # (1, N)
        cache_logits = torch.exp(-1.0 * (self.best_beta - self.best_beta * affinity)) @ self.cache_values
        final_logits = clip_logits + self.best_alpha * cache_logits
        return final_logits  # (1, num_classes)

    def segment_image(self, image, mask):
        """mask領域を切り出して背景黒塗りした PIL.Image を返す"""
        mask = mask.astype(bool)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_array = np.array(image)
        segmented = np.zeros_like(image_array)
        segmented[mask] = image_array[mask]

        black_image = Image.new("RGB", image.size, (0, 0, 0))
        transparency = np.zeros_like(mask, dtype=np.uint8)
        transparency[mask] = 255
        transparency_img = Image.fromarray(transparency, mode='L')

        segmented_pil = Image.fromarray(segmented)
        black_image.paste(segmented_pil, mask=transparency_img)
        return black_image

    @torch.no_grad()
    def relabel_regions(self, image, masks, confidence_threshold=0.02):
        """
        SAM2のマスクを1つずつCLIP + Tip-Adapter 推論
        """
        predictions = []
        masked_regions = []
        for mask in masks:
            seg_img = self.segment_image(image, mask)
            masked_regions.append(seg_img)

            feat = self._encode_image(seg_img)           # shape: (1, dim), float32
            logits = self._tip_adapter_inference(feat)   # (1, num_classes)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            selected_idx = np.argmax(probs)
            selected_class = self.class_names[selected_idx]
            confidence = float(probs[selected_idx])

            if confidence < confidence_threshold:
                selected_class = "unknown"

            # top-k
            top_k = min(3, len(self.class_names))
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_info = []
            for idx_ in top_indices:
                c_name = self.class_names[idx_]
                c_conf = float(probs[idx_])
                top_info.append({'class': c_name, 'probability': c_conf})

            predictions.append({
                'class_name': selected_class,
                'confidence': confidence,
                'top_predictions': top_info
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
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_image(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1) SAM2
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    # 2) GroundingDINO
    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=device
    )
    
    # 3) Tip-Adapter
    tip_relabeler = TipAdapterRelabeler(args.clip_checkpoint, device=device)
    visualizer = ResultVisualizer()

    # 4) 画像読み込み & SAM2 設定
    image_source, image = load_image(args.input_image)
    sam2_predictor.set_image(image_source)

    # 5) GroundingDINO で box 検出
    boxes, _, _ = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    # 6) SAM2 でマスク推論
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 7) Tip-Adapter で領域分類
    predictions, masked_regions = tip_relabeler.relabel_regions(
        image_source,
        masks,
        confidence_threshold=0.02
    )

    # 8) 可視化
    labels = [f"{pred['class_name']} ({pred['confidence']:.2f})" for pred in predictions]
    annotated_frame = visualizer.visualize(image_source, input_boxes, masks, labels)

    output_path = os.path.join(args.output_dir, "relabeled_result.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # 領域画像保存
    crops_dir = os.path.join(args.output_dir, 'cropped_regions')
    os.makedirs(crops_dir, exist_ok=True)
    for i, (region, pred) in enumerate(zip(masked_regions, predictions)):
        out_name = f"region_{i:03d}_{pred['class_name']}_{pred['confidence']:.2f}.png"
        region.save(os.path.join(crops_dir, out_name))

    # JSON 出力
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "image_size": {"width": w, "height": h},
            "predictions": []
        }
        for i, (pred, box, mask_) in enumerate(zip(predictions, input_boxes, masks)):
            rle = mask2rle(mask_)
            results["predictions"].append({
                "region_id": i,
                "class_name": pred["class_name"],
                "confidence": pred["confidence"],
                "bbox": box.tolist(),
                "mask_rle": rle
            })
        json_path = os.path.join(args.output_dir, "detection_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"[INFO] Processing completed! Results saved to {args.output_dir}")

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
