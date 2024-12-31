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
torch.cuda.empty_cache()

def parse_text_prompt(text_prompt):
    """
    テキストプロンプトを個別のクラスラベルに分割する
    
    Args:
        text_prompt (str): ピリオドで区切られたクラスラベル
        
    Returns:
        list: 正規化されたクラスラベルのリスト
    """
    # ピリオドで分割し、空白を削除して正規化
    return [label.strip().lower() for label in text_prompt.split('.') if label.strip()]

def is_valid_label(label, valid_classes):
    """
    ラベルが有効なクラスに属しているかを判定する
    
    Args:
        label (str): 検証するラベル
        valid_classes (list): 有効なクラスラベルのリスト
        
    Returns:
        bool: ラベルが有効な場合True
    """
    # ラベルを小文字に変換して比較
    label_lower = label.lower()
    # 各有効クラスがラベルに完全一致するかチェック
    return any(valid_class == label_lower for valid_class in valid_classes)

def segment_image(image, segmentation_mask):
    """マスクを使用して画像から対象領域を切り出す"""
    segmentation_mask = segmentation_mask.astype(bool)
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    
    # 透明度マスクを使用して背景を黒に
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    
    return black_image

@torch.no_grad()
def classify_with_clip(clip_model, clip_preprocess, images, class_names, device):
    """
    CLIPモデルを使用して画像を分類する
    
    Args:
        clip_model: CLIPモデル
        clip_preprocess: 前処理関数
        images: 分類する画像のリスト
        class_names: 有効なクラス名のリスト
        device: 計算デバイス
        
    Returns:
        tuple: (予測クラス名のリスト, 確信度のリスト)
    """
    if not images:
        return [], []
        
    # 画像の前処理
    preprocessed_images = torch.stack([
        clip_preprocess(image).to(device) for image in images
    ])
    
    # テキストのエンコード
    text_tokens = clip.tokenize(class_names).to(device)
    
    # 特徴量の抽出と正規化
    image_features = clip_model.encode_image(preprocessed_images)
    text_features = clip_model.encode_text(text_tokens)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 類似度の計算
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    similarity = similarity.cpu().numpy()
    
    # 最も確信度の高いクラスを選択
    predicted_classes = []
    confidences = []
    for scores in similarity:
        idx = scores.argmax()
        predicted_classes.append(class_names[idx])
        confidences.append(float(scores[idx]))
    
    return predicted_classes, confidences

def mask_to_polygons(mask):
    """マスクデータをポリゴン形式に変換"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_TC89_KCOS
    )
    
    polygons = []
    for contour in contours:
        # 面積が小さすぎるポリゴンを除外
        if cv2.contourArea(contour) >= 10:
            # Douglas-Peucker アルゴリズムで形状を単純化
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    
    return polygons

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM2 with Selective CLIP Demo", add_help=True)
    # Argument parser
    parser = argparse.ArgumentParser("Grounded SAM2 with CLIP Demo", add_help=True)
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_config", type=str, required=True, help="Path to SAM2 config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="Path to Grounding DINO checkpoint")
    parser.add_argument("--grounded_config", type=str, required=True, help="Path to Grounding DINO config file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device, e.g., 'cpu' or 'cuda'")
    parser.add_argument("--dump_json", action="store_true", help="Dump results to JSON file")
    args = parser.parse_args()

    # 有効なクラスラベルのリストを作成
    valid_classes = parse_text_prompt(args.text_prompt)
    print(f"Valid classes: {valid_classes}")

    # モデルのロード
    device = torch.device(args.device)
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=device
    )
    
    # CLIPモデルのロード
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    clip_model.load_state_dict(torch.load("/mnt/source/Downloads/best_model_total_clip_200.pt", map_location=device))
    clip_model.eval()

    # 画像の読み込みとGrounding DINOの実行
    image_source, image = load_image(args.input_image)
    sam2_predictor.set_image(image_source)
    
    boxes, confidences, initial_labels = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device
    )

    # バウンディングボックスの処理
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # SAM2による予測
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    # マスクの形状を(n, H, W)に変更
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 入力画像をPIL形式に変換
    pil_image = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
    
    # ラベルの有効性をチェックし、必要な場合のみCLIPを適用
    final_labels = []
    final_confidences = []
    images_for_clip = []
    invalid_indices = []
    
    # 初期ラベルをチェック
    for i, label in enumerate(initial_labels):
        if is_valid_label(label, valid_classes):
            # 有効なラベルはそのまま使用
            final_labels.append(label)
            final_confidences.append(float(confidences[i]))
        else:
            # 無効なラベルはCLIPで再分類
            invalid_indices.append(i)
            segmented_image = segment_image(pil_image, masks[i])
            images_for_clip.append(segmented_image)
    
    # 無効なラベルに対してCLIPを適用
    if images_for_clip:
        clip_labels, clip_confidences = classify_with_clip(
            clip_model,
            clip_preprocess,
            images_for_clip,
            valid_classes,
            device
        )
        
        # CLIP結果を適切な位置に挿入
        for idx, (label, confidence) in zip(invalid_indices, zip(clip_labels, clip_confidences)):
            final_labels.insert(idx, label)
            final_confidences.insert(idx, confidence)

    # 表示用のラベル作成
    display_labels = [
        f"{label} {conf:.2f}" for label, conf in zip(final_labels, final_confidences)
    ]

    # 可視化
    img = cv2.imread(args.input_image)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=np.array(list(range(len(final_confidences))))
    )

    # アノテーション処理
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=display_labels
    )
    annotated_frame_with_mask = mask_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    # 画像の保存
    cv2.imwrite(os.path.join(args.output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)
    cv2.imwrite(os.path.join(args.output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame_with_mask)

    # JSON出力
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "annotations": []
        }
        
        # 各検出結果の詳細情報を保存
        for i, (mask, box, label, confidence) in enumerate(zip(
            masks, input_boxes, final_labels, final_confidences
        )):
            polygons = mask_to_polygons(mask)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            
            annotation = {
                "class_name": label,
                "bbox": box.tolist(),
                "segmentation": {
                    "polygons": polygons,
                    "rle": mask_rle
                },
                "score": float(confidence),
                "label_source": "SAM2" if is_valid_label(label, valid_classes) else "CLIP"
            }
            
            results["annotations"].append(annotation)
        
        results.update({
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
            "valid_classes": valid_classes
        })
        
        with open(os.path.join(args.output_dir, "grounded_sam2_results.json"), "w") as f:
            json.dump(results, f, indent=4)