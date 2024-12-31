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
import torch.nn.functional as F

# GPUメモリの最適化のためにキャッシュをクリア
torch.cuda.empty_cache()

# マルチモーダルな分類のためのCLIP関連のインポート
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def calculate_box_area(box):
    """バウンディングボックスの面積を計算する関数
    
    Args:
        box (list): [x1, y1, x2, y2]形式のバウンディングボックス座標
    
    Returns:
        float: バウンディングボックスの面積
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def calculate_iou(box1, box2):
    """2つのバウンディングボックス間のIoU（Intersection over Union）を計算する関数
    
    Args:
        box1 (list): 1つ目のバウンディングボックス [x1, y1, x2, y2]
        box2 (list): 2つ目のバウンディングボックス [x1, y1, x2, y2]
    
    Returns:
        float: 2つのボックス間のIoU値（0-1の範囲）
    """
    # 交差領域の座標を計算
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 交差領域の面積を計算
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 各ボックスの面積を計算
    box1_area = calculate_box_area(box1)
    box2_area = calculate_box_area(box2)
    
    # 和集合の面積を計算
    union_area = box1_area + box2_area - intersection_area
    
    # IoUを計算して返す（0での除算を防ぐ）
    return intersection_area / union_area if union_area > 0 else 0

def filter_overlapping_boxes(boxes, masks, class_names, confidences, iou_threshold=0.5):
    """重複するバウンディングボックスをフィルタリングし、各クラスで最小のものを保持する関数
    
    Args:
        boxes (np.ndarray): バウンディングボックスの配列
        masks (np.ndarray): セグメンテーションマスクの配列
        class_names (list): クラス名のリスト
        confidences (list): 信頼度スコアのリスト
        iou_threshold (float): IoUの閾値（デフォルト: 0.5）
    
    Returns:
        tuple: フィルタリング後の(boxes, masks, class_names, confidences)
    """
    if len(boxes) == 0:
        return boxes, masks, class_names, confidences
    
    # 入力をNumPy配列に変換して処理を効率化
    boxes = np.array(boxes)
    masks = np.array(masks)
    class_names = np.array(class_names)
    confidences = np.array(confidences)
    
    # すべてのボックスの面積を計算
    areas = np.array([calculate_box_area(box) for box in boxes])
    
    # 面積でソート（昇順）して小さいボックスを優先
    sorted_indices = np.argsort(areas)
    
    # 保持するインデックスのリスト
    keep_indices = []
    
    # クラスごとに処理を行う
    for class_name in np.unique(class_names):
        # 現在のクラスのインデックスを取得
        class_indices = np.where(class_names == class_name)[0]
        
        # クラスのインデックスを面積でソート
        class_indices = np.array([idx for idx in sorted_indices if idx in class_indices])
        
        if len(class_indices) == 0:
            continue
            
        # 重複チェックと最小ボックスの保持
        current_keep_indices = []
        for i, idx1 in enumerate(class_indices):
            keep = True
            for idx2 in current_keep_indices:
                if calculate_iou(boxes[idx1], boxes[idx2]) > iou_threshold:
                    keep = False
                    break
            if keep:
                current_keep_indices.append(idx1)
        
        keep_indices.extend(current_keep_indices)
    
    # インデックスをソートして元の順序を維持
    keep_indices = sorted(keep_indices)
    
    # フィルタリングされた結果を返す
    return (
        boxes[keep_indices],
        masks[keep_indices],
        class_names[keep_indices],
        confidences[keep_indices]
    )

def single_mask_to_rle(mask):
    """セグメンテーションマスクをRLE（Run-Length Encoding）形式に変換する関数
    
    Args:
        mask (np.ndarray): バイナリマスク
    
    Returns:
        dict: RLE形式のマスクデータ
    """
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def segment_image(image, segmentation_mask):
    """画像をセグメンテーションマスクに基づいて切り出す関数
    
    Args:
        image (PIL.Image): 入力画像
        segmentation_mask (np.ndarray): セグメンテーションマスク
    
    Returns:
        PIL.Image: マスクされた画像
    """
    # マスクをブール型に変換
    segmentation_mask = segmentation_mask.astype(bool)
    
    # 画像とマスクの処理
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    
    # 透明な背景を持つ新しい画像を作成
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    
    # マスクを適用して最終的な画像を作成
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

@torch.no_grad()
def retrieve_clip_scores(elements: list, search_texts: list, model, preprocess, device):
    """CLIPモデルを使用して画像とテキストの類似度スコアを計算する関数
    
    Args:
        elements (list): 画像のリスト
        search_texts (list): テキストプロンプトのリスト
        model: CLIPモデル
        preprocess: CLIP前処理関数
        device: 計算デバイス
    
    Returns:
        np.ndarray: 類似度スコア
    """
    # 画像の前処理とバッチ化
    preprocessed_images = torch.stack([preprocess(image).to(device) for image in elements])
    
    # テキストのトークン化
    tokenized_text = clip.tokenize(search_texts).to(device)
    
    # 特徴量の抽出と正規化
    image_features = model.encode_image(preprocessed_images)
    text_features = model.encode_text(tokenized_text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 類似度スコアの計算
    similarity = image_features @ text_features.T
    probs = similarity.softmax(dim=-1)
    
    return probs.cpu().numpy()

if __name__ == "__main__":
    # コマンドライン引数の設定
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
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--dump_json", action="store_true", help="Dump results to JSON file")
    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # SAM2モデルの構築
    print("Loading SAM2 model...")
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Grounding DINOモデルの構築
    print("Loading Grounding DINO model...")
    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=args.device
    )

    # 画像の読み込みとSAM2の準備
    print("Loading and processing image...")
    image_source, image = load_image(args.input_image)
    sam2_predictor.set_image(image_source)

    # Grounding DINOモデルによる初期検出
    print("Running Grounding DINO detection...")
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device
    )

    # 画像サイズに合わせてボックスのスケーリング
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # SAM2による予測
    print("Running SAM2 segmentation...")
    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            # Ampere GPUs用にTF32を有効化
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    # マスクの形状を(n, H, W)に変換
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 初期ラベルとconfidencesの準備
    confidences = confidences.numpy().tolist()
    initial_class_names = [label.strip() for label in args.text_prompt.strip().split('.') if label.strip()]
    print(f"Initial class labels: {initial_class_names}")
    
    # 重複ボックスのフィルタリング
    print("Filtering overlapping boxes...")
    filtered_boxes, filtered_masks, filtered_initial_labels, filtered_confidences = filter_overlapping_boxes(
        input_boxes, 
        masks,
        [initial_class_names[0]] * len(confidences),  # 仮のクラス名を使用
        confidences,
        iou_threshold=0.5
    )

# CLIPモデルのセットアップと準備
    print("Loading CLIP model and finetuned weights...")
    device = torch.device(args.device)
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    
    # ラベル情報の読み込みと処理
    print("Loading class labels from training data...")
    with open('/mnt/source/Downloads/labels_total.json', 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    # 除外するクラスを定義
    excluded_classes = {
        "Right of way at next intersection",  # 誤検出の可能性が高いクラス
        "Roundabout mandatory",
        "No passing",
        # 必要に応じて他のクラスを追加
    }
    
    # 除外クラスをフィルタリングしてユニークなクラスラベルを抽出
    finetuned_class_names = list(set(
        item['label'] for item in labels_data 
        if item['label'] not in excluded_classes
    ))
    print(f"Found {len(finetuned_class_names)} unique classes: {finetuned_class_names}")
    
    # 学習済みの重みを読み込み
    print("\nLoading finetuned weights...")
    checkpoint_path = '/mnt/source/Downloads/best_model_total_clip_200.pt'
    clip_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    clip_model.eval()

    # 画像の前処理とマスク領域の切り出し
    pil_image = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
    cropped_boxes = [segment_image(pil_image, mask) for mask in filtered_masks]

    @torch.no_grad()
    def retrieve_finetuned_scores(images, class_names, model, preprocess, device):
        preprocessed_images = torch.stack([preprocess(image).to(device) for image in images])
        tokenized_text = clip.tokenize(class_names, truncate=True).to(device)
        
        with torch.cuda.amp.autocast():
            image_features = model.encode_image(preprocessed_images)
            text_features = model.encode_text(tokenized_text)
            
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
            
            similarity = (image_features @ text_features.T) / 0.07
            probs = similarity.softmax(dim=-1)
        
        return probs.cpu().numpy()

    # 検出領域の分類を実行
    print("\nClassifying detected regions...")
    probs = retrieve_finetuned_scores(
        cropped_boxes,
        finetuned_class_names,
        clip_model,
        clip_preprocess,
        device
    )

    # 分類結果の処理
    final_class_names = []
    final_confidences = []
    for prob in probs:
        max_prob_index = np.argmax(prob)
        final_class_names.append(finetuned_class_names[max_prob_index])
        final_confidences.append(float(prob[max_prob_index]))

    # ラベルの生成
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(final_class_names, final_confidences)
    ]
    
    # 検出結果の可視化
    print("Visualizing results...")
    img = cv2.imread(args.input_image)
    detections = sv.Detections(
        xyxy=filtered_boxes,
        mask=filtered_masks.astype(bool),
        class_id=np.array(list(range(len(final_confidences))))
    )

    # バウンディングボックスとラベルのアノテーション
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # 結果の保存
    print("Saving results...")
    output_path = os.path.join(args.output_dir, "detection_results.jpg")
    cv2.imwrite(output_path, annotated_frame)

    # JSON形式での結果保存
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "annotations": [
                {
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": box.tolist(),
                    "mask": single_mask_to_rle(mask)
                }
                for class_name, conf, box, mask in zip(
                    final_class_names,
                    final_confidences,
                    filtered_boxes,
                    filtered_masks
                )
            ],
            "finetuned_classes": finetuned_class_names
        }
        
        with open(os.path.join(args.output_dir, "detection_results.json"), "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print("Processing completed successfully!")