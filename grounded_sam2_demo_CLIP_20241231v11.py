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

class TipAdapterRelabeler:
    """
    学習済み Tip-Adapter チェックポイント (cache_keys, cache_values, clip_weights, best_alpha, best_beta etc.)
    を読み込み、SAM2 で切り出した領域画像を CLIP + Tip-Adapter で再ラベリングする。
    """
    def __init__(self, tip_adapter_checkpoint, device='cuda'):
        self.device = device
        # 1) CLIP モデルをロード
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.clip_model = self.clip_model.eval().float()

        # 2) 学習済みキャッシュをロード
        ckpt = torch.load(tip_adapter_checkpoint, map_location=self.device)
        print(f"[INFO] Loaded Tip-Adapter cache from: {tip_adapter_checkpoint}")

        # 3) キャッシュ等を float32 にして同じデバイス上に乗せる
        self.cache_keys   = ckpt['cache_keys'].float().to(self.device)       # shape: (dim, N)
        self.cache_values = ckpt['cache_values'].float().to(self.device)     # shape: (N, C)
        self.clip_weights = ckpt['clip_weights'].float().to(self.device)     # shape: (dim, C)
        self.best_alpha   = ckpt['best_alpha']
        self.best_beta    = ckpt['best_beta']

        # 4) クラス名リスト
        if 'selected_labels' in ckpt:
            self.class_names = ckpt['selected_labels']
        else:
            # フォールバック
            self.class_names = selected_labels

    @torch.no_grad()
    def _encode_image(self, pil_image: Image.Image):
        """
        領域画像 (PIL) を CLIP encode_image でベクトル化し、L2正規化して返す (shape: (1, dim))
        """
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        feat = self.clip_model.encode_image(img_tensor).float()
        feat = F.normalize(feat, p=2, dim=-1)
        return feat

    @torch.no_grad()
    def _tip_adapter_inference(self, image_features):
        """
        Tip-Adapter のロジック:
          clip_logits = 100.0 * (image_features @ clip_weights)
          affinity = image_features @ cache_keys
          cache_logits = exp(-beta * (1 - affinity)) @ cache_values
          final_logits = clip_logits + alpha * cache_logits
        """
        # 1) Zero-shot CLIP ロジット
        clip_logits = 100.0 * (image_features @ self.clip_weights)  # (1, num_classes)

        # 2) キャッシュとの類似度
        affinity = image_features @ self.cache_keys                 # (1, N)
        # 論文中: exp(- beta * (1 - affinity)) だが、公式実装では
        #        exp(-1 * (beta - beta * affinity)) と同値
        cache_logits = torch.exp(-1.0 * (self.best_beta - self.best_beta * affinity)) @ self.cache_values

        # 3) 合算
        final_logits = clip_logits + self.best_alpha * cache_logits  # (1, num_classes)
        return final_logits

    def segment_image(self, image, mask):
        """
        mask(2D) で領域を切り出し、背景黒塗りした PIL.Image を返す
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        mask_bool = mask.astype(bool)
        arr = np.array(image)
        seg = np.zeros_like(arr)
        seg[mask_bool] = arr[mask_bool]

        black_img = Image.new("RGB", image.size, (0, 0, 0))
        transp = np.zeros_like(mask_bool, dtype=np.uint8)
        transp[mask_bool] = 255
        transp_img = Image.fromarray(transp, mode='L')

        seg_pil = Image.fromarray(seg)
        black_img.paste(seg_pil, mask=transp_img)
        return black_img
    
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
        masks: (num_boxes, H, W)
        各マスク領域を CLIP + Tip-Adapter でラベリング → 予測クラス名とconfidence, さらに上位クラスの情報を返す
        """
        predictions = []
        masked_regions = []

        for mask in masks:
            # 1) 領域画像: bounding box クロップ
            cropped_pil = self.crop_masked_region(image, mask)
            masked_regions.append(cropped_pil)

            # 2) CLIP エンコード + Tip-Adapter 推論
            feat = self._encode_image(cropped_pil)             # (1, dim)
            logits = self._tip_adapter_inference(feat)     # (1, num_classes)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            # 最上位クラスの取得
            top_idx = np.argmax(probs)
            top_class = self.class_names[top_idx]
            confidence = float(probs[top_idx])

            if confidence < confidence_threshold:
                top_class = "unknown"

            # 上位3クラスも取得
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
        """
        SAM2 + Tip-Adapter で推定した結果を可視化 (box + mask + label)
        """
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks.astype(bool),
            class_id=np.arange(len(boxes))
        )
        annotated = image.copy()

        # 1) マスクの可視化
        mask_annotator = sv.MaskAnnotator(opacity=0.3)
        annotated = mask_annotator.annotate(annotated, detections=detections)

        # 2) バウンディングボックスの可視化
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated = box_annotator.annotate(annotated, detections=detections)

        # 3) ラベル (クラス名) の可視化
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
        annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)

        return annotated

def mask2rle(mask):
    """
    マスクを COCO API 用の RLE に変換
    """
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def process_image(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1) SAM2 モデルをロードして予測器を作成
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # 2) GroundingDINO モデルをロード
    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=device
    )

    # 3) 学習済み Tip-Adapter (cache) を読み込み
    tip_relabeler = TipAdapterRelabeler(args.clip_checkpoint, device=device)
    visualizer = ResultVisualizer()

    # 4) 入力画像を読み込み＆SAM2 にセット
    image_source, image = load_image(args.input_image)
    sam2_predictor.set_image(image_source)

    # 5) GroundingDINO で物体検出 (box)
    boxes, _, _ = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device
    )
    h, w, _ = image_source.shape
    # (cx, cy, w, h) → (x1, y1, x2, y2)
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
        masks = masks.squeeze(1)  # shape: (N, H, W)

    # 7) 領域ごとに Tip-Adapter でラベルを推定
    predictions, masked_regions = tip_relabeler.relabel_regions(image_source, masks, confidence_threshold=0.02)

    # 8) 可視化
    labels = [f"{pred['class_name']} ({pred['confidence']:.2f})" for pred in predictions]
    annotated = visualizer.visualize(image_source, input_boxes, masks, labels)

    # 9) 結果保存
    out_img_path = os.path.join(args.output_dir, "relabeled_result.jpg")
    cv2.imwrite(out_img_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved annotated result to: {out_img_path}")

    # 領域画像を保存
    crops_dir = os.path.join(args.output_dir, "cropped_regions")
    os.makedirs(crops_dir, exist_ok=True)
    for i, (region, pred) in enumerate(zip(masked_regions, predictions)):
        filename = f"region_{i:03d}_{pred['class_name']}_{pred['confidence']:.2f}.png"
        region.save(os.path.join(crops_dir, filename))

    # JSON 出力 (オプション)
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
        print(f"[INFO] Saved detection results JSON to: {json_path}")

    print(f"[INFO] Done. Results saved to: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser("Grounded SAM2 + Tip-Adapter Demo", add_help=True)
    parser.add_argument("--sam2_checkpoint", required=True, type=str, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_config", required=True, type=str, help="Path to SAM2 config file")
    parser.add_argument("--grounded_checkpoint", required=True, type=str, help="Path to GroundingDINO checkpoint")
    parser.add_argument("--grounded_config", required=True, type=str, help="Path to GroundingDINO config file")
    parser.add_argument("--clip_checkpoint", required=True, type=str, help="Path to tip_adapter_cache.pt (Tip-Adapter)")
    parser.add_argument("--input_image", required=True, type=str, help="Input image path")
    parser.add_argument("--text_prompt", required=True, type=str, help="Text prompt for GroundingDINO")
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="outputs_tip_adapter")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dump_json", action="store_true", help="If set, save detection results as JSON")
    args = parser.parse_args()

    process_image(args)

if __name__ == "__main__":
    main()
