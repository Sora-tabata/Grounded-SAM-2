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
import torch.nn as nn
from PIL import Image

# --- SAM2 & GroundingDINO 関連 ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# ================== クラスラベル (学習時と同じ) ==================
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

class TipAdapterFRelabeler:
    """
    Fine-tuning されたアダプタ (adapter.weight) を含む Tip-Adapter-F をロードし、
    SAM2 の領域画像を CLIP encode + adapter に通すロジックで再ラベリングするクラス。
    """
    def __init__(self, tip_adapter_checkpoint, adapter_weight_path, device='cuda'):
        """
        Args:
            tip_adapter_checkpoint: 学習時に保存した tip_adapter_cache.pt 
                （cache_keys, cache_values, clip_weights, alpha, beta, etc.）
            adapter_weight_path: Fine-tuning 後に保存された best_F_{shots}.pt のパス（adapter.weight）
        """
        self.device = torch.device(device)

        # 1) CLIP モデル
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.clip_model.eval().float()

        # 2) cache などをロード
        ckpt = torch.load(tip_adapter_checkpoint, map_location=self.device)
        print(f"[INFO] Loaded Tip-Adapter cache from: {tip_adapter_checkpoint}")

        # 3) 各テンソルを float32 + device に
        self.cache_keys   = ckpt['cache_keys'].float().to(self.device)    # shape: (dim, N)
        self.cache_values = ckpt['cache_values'].float().to(self.device)  # shape: (N, C)
        self.clip_weights = ckpt['clip_weights'].float().to(self.device)  # shape: (dim, C)
        self.best_alpha   = ckpt['best_alpha']
        self.best_beta    = ckpt['best_beta']
        # ラベルリスト
        if 'selected_labels' in ckpt:
            self.class_names = ckpt['selected_labels']
        else:
            self.class_names = selected_labels

        # 4) adapter を再構築
        dim = self.cache_keys.shape[0]   # (dim, N) -> dim
        N   = self.cache_keys.shape[1]   # (dim, N) -> N
        self.adapter = nn.Linear(dim, N, bias=False).to(self.device)
        
        # 5) Fine-tuning 後の weight を読み込み
        best_weight = torch.load(adapter_weight_path, map_location=self.device)
        self.adapter.weight = nn.Parameter(best_weight.clone().float().to(self.device))

        print(f"[INFO] Loaded fine-tuned adapter weight from: {adapter_weight_path}")

    @torch.no_grad()
    def _encode_image(self, pil_image: Image.Image):
        """領域画像を CLIP encode して L2 正規化"""
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        feat = self.clip_model.encode_image(img_tensor).float()
        feat = F.normalize(feat, p=2, dim=-1)
        return feat  # shape: (1, dim)

    @torch.no_grad()
    def _tip_adapter_f_inference(self, image_features):
        """
        Tip-Adapter-F のロジック:
          - adapter(image_features) = affinity
          - cache_logits = exp(-1*(beta - beta*affinity)) @ cache_values
          - clip_logits = 100. * (image_features @ clip_weights)
          - final_logits = clip_logits + alpha * cache_logits
        """
        # 1) adapter
        affinity = self.adapter(image_features)  # (1, N)
        cache_logits = torch.exp(-1.0 * (self.best_beta - self.best_beta * affinity)) @ self.cache_values

        # 2) zero-shot clip
        clip_logits = 100.0 * (image_features @ self.clip_weights)  # (1, C)

        # 3) combine
        final_logits = clip_logits + self.best_alpha * cache_logits  # (1, C)
        return final_logits

    def crop_masked_region(self, image, mask):
        """
        mask領域を bounding box でクロップした画像(PIL)を返す.
        黒塗りではなく、(ymin:ymax, xmin:xmax) を切り出した領域だけを取得。
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        mask_bool = mask.astype(bool)
        # mask の True 画素が無ければそのまま全体(安全策)
        if not mask_bool.any():
            return image
        
        arr = np.array(image)
        # マスクの nonzero を見て bounding box を計算
        ys, xs = np.where(mask_bool)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        # bounding box で画像を切り出し
        cropped_arr = arr[ymin:ymax+1, xmin:xmax+1, :]
        cropped_pil = Image.fromarray(cropped_arr)
        return cropped_pil

    @torch.no_grad()
    def relabel_regions(self, image, masks, confidence_threshold=0.02):
        """
        SAM2のマスクを順番に CLIP encode -> adapter -> Tip-Adapter-F のロジックでクラス推定
        """
        predictions = []
        masked_regions = []
        for mask in masks:
            # 領域画像を、(ymin:ymax, xmin:xmax) でクロップして取得
            cropped_pil = self.crop_masked_region(image, mask)
            masked_regions.append(cropped_pil)

            # 特徴量
            feat = self._encode_image(cropped_pil)  # (1, dim)
            logits = self._tip_adapter_f_inference(feat)  # (1, C)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            top_idx = np.argmax(probs)
            top_class = self.class_names[top_idx]
            confidence = float(probs[top_idx])

            if confidence < confidence_threshold:
                top_class = "unknown"

            # top-k
            top_k = min(3, len(self.class_names))
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_pred_list = []
            for idx_ in top_indices:
                top_pred_list.append({
                    'class': self.class_names[idx_],
                    'probability': float(probs[idx_])
                })

            predictions.append({
                'class_name': top_class,
                'confidence': confidence,
                'top_predictions': top_pred_list
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
        annotated = image.copy()

        mask_annotator = sv.MaskAnnotator(opacity=0.3)
        annotated = mask_annotator.annotate(annotated, detections=detections)

        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated = box_annotator.annotate(annotated, detections=detections)

        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
        annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
        return annotated

def mask2rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def process_image(args):
    # 1) SAM2, GroundingDINO のロード
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=args.device
    )

    # 2) Tip-Adapter-F (cache + fine-tuned weight) ロード
    tip_relabeler = TipAdapterFRelabeler(
        tip_adapter_checkpoint=args.clip_checkpoint,
        adapter_weight_path=args.adapter_weight,
        device=args.device
    )
    visualizer = ResultVisualizer()

    # 3) 入力画像ロード
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    image_source, image = load_image(args.input_image)
    sam2_predictor.set_image(image_source)

    # 4) GroundingDINO で BBox 検出
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

    # 5) SAM2 でマスク推論
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 6) 領域を Tip-Adapter-F で再ラベリング (bounding box crop)
    predictions, masked_regions = tip_relabeler.relabel_regions(
        image_source,
        masks,
        confidence_threshold=0.05  # 少し高めにして「unknown」判定を厳しく
    )

    # 7) 可視化
    labels = [f"{pred['class_name']} ({pred['confidence']:.2f})" for pred in predictions]
    annotated_frame = visualizer.visualize(image_source, input_boxes, masks, labels)

    # 8) 保存
    out_img_path = os.path.join(args.output_dir, "relabeled_result.jpg")
    cv2.imwrite(out_img_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved annotated image to {out_img_path}")

    # 領域画像保存
    crops_dir = os.path.join(args.output_dir, "cropped_regions")
    os.makedirs(crops_dir, exist_ok=True)
    for i, (region, pred) in enumerate(zip(masked_regions, predictions)):
        filename = f"region_{i:03d}_{pred['class_name']}_{pred['confidence']:.2f}.png"
        region.save(os.path.join(crops_dir, filename))

    # 9) JSON 出力 (オプション)
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "image_size": {"width": w, "height": h},
            "predictions": []
        }
        for i, (pred, box, mk) in enumerate(zip(predictions, input_boxes, masks)):
            rle = mask2rle(mk)
            results["predictions"].append({
                "region_id": i,
                "class_name": pred["class_name"],
                "confidence": pred["confidence"],
                "bbox": box.tolist(),
                "mask_rle": rle
            })
        json_out = os.path.join(args.output_dir, "detection_results.json")
        with open(json_out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Saved detection results JSON to: {json_out}")

    print(f"[INFO] Done. Results saved in {args.output_dir}")

def main():
    parser = argparse.ArgumentParser("SAM2 + Tip-Adapter-F Demo", add_help=True)
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--sam2_config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--grounded_config", type=str, required=True)

    parser.add_argument("--clip_checkpoint", type=str, required=True, 
                        help="Path to tip_adapter_cache.pt (cache_keys, cache_values, clip_weights, alpha, beta, etc.)")
    parser.add_argument("--adapter_weight", type=str, required=True,
                        help="Path to best_F_{shots}.pt (fine-tuned adapter.weight)")

    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="outputs_tip_adapter_f")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dump_json", action="store_true")
    args = parser.parse_args()

    process_image(args)

if __name__ == "__main__":
    main()
