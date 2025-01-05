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
from PIL import Image
import torch.nn.functional as F

# ===== SAM2 & GroundingDINO 関連 =====
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# ===== adapters.py の中身を直接 or 関数として利用 =====
from adapters import ADAPTER, load_clip_to_cpu, CustomCLIP
from dassl.config import get_cfg_default
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_checkpoint

import clip
from train_v12_total import extend_cfg_for_adapter  # 既存の関数を流用

# ================== クラスラベル (train_v12_total.py と同様) ==================
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


# ------------------------------
# WBF 用 IoU 関数
# ------------------------------
def iou(b1, b2):
    """
    b1, b2: [x1, y1, x2, y2]
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0
    return inter / union


# ------------------------------
# 指定の WBF 関数
#   bboxes: list of [x1, y1, x2, y2]
#   scores: list of confidence (float)
#   labels: list of str (クラス名を保持しておきたい場合)
# ------------------------------
def wbf(bboxes: list, scores: list, labels: list,
        iou_threshold: float, n: int):
    """
    bboxes, scores, labels はすべて同じ長さを想定。
    WBF後、それぞれ同じ長さの fused_bboxes, fused_scores, fused_labels を返す
    """
    lists, fusions, confidences, label_clusters = [], [], [], []

    # スコアの高い順にインデックスを並び替える
    indexes = sorted(range(len(bboxes)), key=scores.__getitem__)[::-1]

    for i in indexes:
        new_fusion = True
        for j in range(len(fusions)):
            if iou(bboxes[i], fusions[j]) > iou_threshold:
                # 同じクラスターにまとめる
                lists[j].append(bboxes[i])
                confidences[j].append(scores[i])
                label_clusters[j].append(labels[i])
                # 重み付き平均で BBox を更新
                wsum = sum(confidences[j])
                fusions[j] = (
                    sum([l[0] * c for l, c in zip(lists[j], confidences[j])]) / wsum,
                    sum([l[1] * c for l, c in zip(lists[j], confidences[j])]) / wsum,
                    sum([l[2] * c for l, c in zip(lists[j], confidences[j])]) / wsum,
                    sum([l[3] * c for l, c in zip(lists[j], confidences[j])]) / wsum,
                )
                new_fusion = False
                break

        if new_fusion:
            lists.append([bboxes[i]])
            confidences.append([scores[i]])
            fusions.append(bboxes[i])
            label_clusters.append([labels[i]])

    # 各クラスタの confidence
    fused_scores = [
        (sum(c) / len(c)) * (min(n, len(c)) / n) for c in confidences
    ]

    # ラベルは最もスコアの高いものを代表とするか、あるいは結合して返すか等、好みで実装
    # ここではスコア最大のラベルにする
    fused_labels = []
    for c, lc in zip(confidences, label_clusters):
        idx_max = np.argmax(c)
        fused_labels.append(lc[idx_max])

    return fusions, fused_scores, fused_labels


# ----------------------------------------------------------------------------
# 1) 学習済みアダプタ (tar-10, tar-1など) を推論用にロードするためのラッパ
# ----------------------------------------------------------------------------
class InferenceAdapterTrainer(ADAPTER):
    """
    adapters.py の ADAPTER を継承し、推論に必要な部分だけ初期化＆利用するための簡易クラス。
    """
    def __init__(self, adapter_init_mode, device='cuda'):
        self.cfg = get_cfg_default()
        extend_cfg_for_adapter(self.cfg)
        self.cfg.defrost()
        # 好みで細かい設定
        self.cfg.MODEL.BACKBONE.NAME = "ViT-B/32"
        self.cfg.TRAINER.ADAPTER.INIT = adapter_init_mode
        self.cfg.TRAINER.ADAPTER.PREC = "fp32"
        self.cfg.TRAINER.NAME = "ADAPTER"
        self.cfg.DATASET.NAME = "CIFAR10"  # ダミー
        self.cfg.CLASSNAMES = selected_labels
        self.cfg.OUTPUT_DIR = "./checkpoints_infer_tmp"
        self.cfg.freeze()

        self.classnames = selected_labels
        self.lab2cname = {i: cname for i, cname in enumerate(self.classnames)}

        super().__init__(self.cfg)

        self.device = torch.device(device)
        self.start_epoch = 0
        self.max_epoch = 1  # 学習はしないので1に

    def build_data_loader(self):
        pass

    def train(self):
        pass

    def build_model(self):
        pass

    def load_model_from_dir(self, ckpt_dir, epoch=None):
        print(f"[INFO] Load model from dir: {ckpt_dir}, epoch={epoch}")
        super().load_model(ckpt_dir, self.cfg, epoch=epoch)  # 親の load_model()

    def build_inference_model(self):
        import clip
        cfg = self.cfg
        clip_model = load_clip_to_cpu(cfg)
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
        clip_model.float()

        self.classnames = selected_labels
        self.lab2cname = {i: cname for i, cname in enumerate(self.classnames)}

        clip_model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.preprocess_fn = preprocess
        self.clip_model = clip_model

        self.model = CustomCLIP(cfg, self.classnames, clip_model)
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)
        self.model.to(self.device)
        self.model = self.model.float()

        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter_inference", self.model.adapter, self.optim, self.sched)

    @torch.no_grad()
    def infer_region(self, region_tensor: torch.Tensor) -> torch.Tensor:
        self.set_model_mode("eval")
        logits = self.model.forward_features(region_tensor.to(self.device))
        return logits


# ----------------------------------------------------------------------------
# 2) 上記ラッパを使って、CLIP-Adapter / Tip-Adapter / Tip-Adapter-F / CrossModal を
#    それぞれロード・推論するクラス
# ----------------------------------------------------------------------------
class AdapterRelabeler:
    def __init__(self, adapter_init_mode: str, ckpt_dir: str, epoch_num: int, device='cuda'):
        self.trainer = InferenceAdapterTrainer(adapter_init_mode, device=device)
        self.trainer.build_inference_model()
        self.trainer.load_model_from_dir(ckpt_dir, epoch=epoch_num)
        self.class_names = selected_labels
        self.device = torch.device(device)

    @property
    def preprocess_fn(self):
        return self.trainer.preprocess_fn

    @property
    def clip_model(self):
        return self.trainer.clip_model

    @torch.no_grad()
    def relabel_regions(self, pil_images, confidence_threshold=0.02):
        """
        マスク領域を切り出した PIL 画像のリスト pil_images に対してアダプタ推論を行い、
        class_name と confidence を返す。
        """
        predictions = []
        for pil_img in pil_images:
            img_tensor = self.trainer.preprocess_fn(pil_img).unsqueeze(0).to(self.device)
            # 直接 encode_image して正規化
            clip_features = self.trainer.model.clip_model.encode_image(img_tensor)
            clip_features = F.normalize(clip_features, p=2, dim=-1)
            # adapter込み
            logits = self.trainer.model.forward_features(clip_features)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            top_idx = np.argmax(probs)
            top_class = self.class_names[top_idx]
            confidence = float(probs[top_idx])
            if confidence < confidence_threshold:
                top_class = "unknown"

            predictions.append({
                'class_name': top_class,
                'confidence': confidence
            })
        return predictions


# ----------------------------------------------------------------------------
# 可視化
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# メイン処理
#   text_prompt (アダプタ有) と text_prompt_new (アダプタ無) を使い、
#   4種類のアダプタについてそれぞれ WBF で融合し、画像＋JSON を出力
# ----------------------------------------------------------------------------
def process_image(args):
    device = torch.device(args.device)
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=device
    )
    visualizer = ResultVisualizer()

    # 1) 入力画像読み込み + size
    os.makedirs(args.output_dir, exist_ok=True)
    image_source, image_pil = load_image(args.input_image)
    sam2_predictor.set_image(image_source)
    h, w, _ = image_source.shape

    # ------------------------------------------------------
    # 2) text_prompt 用: GroundingDINO で BBox, Score, SAM2マスクを取得
    # ------------------------------------------------------
    boxes_prompt, logits_prompt, phrases_prompt = predict(
        model=grounding_model,
        image=image_pil,
        caption=args.text_prompt.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device
    )
    boxes_prompt = boxes_prompt * torch.tensor([w, h, w, h], device=boxes_prompt.device)
    input_boxes_prompt = box_convert(boxes=boxes_prompt, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
    scores_prompt = logits_prompt.sigmoid().cpu().numpy()

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
        masks_prompt, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes_prompt,
            multimask_output=False
        )
    if masks_prompt.ndim == 4:
        masks_prompt = masks_prompt.squeeze(1)

    # ------------------------------------------------------
    # 3) text_prompt_new 用: GroundingDINO で BBox, Score, SAM2マスクを取得
    #    ※今回「SAM2でついたクラスラベル」は実際には GroundingDINO の phrases を流用
    # ------------------------------------------------------
    boxes_new, logits_new, phrases_new = predict(
        model=grounding_model,
        image=image_pil,
        caption=args.text_prompt_new.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device
    )
    boxes_new = boxes_new * torch.tensor([w, h, w, h], device=boxes_new.device)
    input_boxes_new = box_convert(boxes=boxes_new, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
    scores_new = logits_new.sigmoid().cpu().numpy()

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
        masks_new, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes_new,
            multimask_output=False
        )
    if masks_new.ndim == 4:
        masks_new = masks_new.squeeze(1)

    # phrases_new からラベルを作成 (scoreも付けておく)
    # 例: "stop sign" (0.84) のようにラベル化する
    no_adapter_labels = [
        f"{ph} ({sc:.2f})" for ph, sc in zip(phrases_new, scores_new)
    ]
    # ------------------------------------------------------
    # アダプタ4種類をループで処理し、それぞれ融合結果を保存する
    # ------------------------------------------------------
    adapter_info_list = [
        {
            "name": "clip",
            "adapter_init": "ClipA",
            "ckpt_dir": args.clip_adapter_dir,
            "epoch": args.clip_adapter_epoch
        },
        {
            "name": "tip",
            "adapter_init": "TipA",
            "ckpt_dir": args.tip_adapter_dir,
            "epoch": args.tip_adapter_epoch
        },
        {
            "name": "tip_f",
            "adapter_init": "TipF",
            "ckpt_dir": args.tip_adapter_f_dir,
            "epoch": args.tip_adapter_f_epoch
        },
        {
            "name": "cm",
            "adapter_init": "CrossModalA",
            "ckpt_dir": args.cross_modal_dir,
            "epoch": args.cross_modal_epoch
        }
    ]

    for adapter_info in adapter_info_list:
        adapter_name = adapter_info["name"]  # "clip", "tip", "tip_f", "cm"

        # 4) アダプタをロード
        print(f"\n[INFO] Processing adapter: {adapter_name}")
        relabeler = AdapterRelabeler(
            adapter_init_mode=adapter_info["adapter_init"],
            ckpt_dir=adapter_info["ckpt_dir"],
            epoch_num=adapter_info["epoch"],
            device=device
        )

        # 5) text_prompt 側だけアダプタで再ラベリング
        masked_pil_list = []
        for mk in masks_prompt:
            mk_bool = mk.astype(bool)
            if not mk_bool.any():
                # 空マスクなら、画像全体を切り出すなど適宜
                cropped_pil = Image.fromarray(image_source)
            else:
                ys, xs = np.where(mk_bool)
                ymin, ymax = ys.min(), ys.max()
                xmin, xmax = xs.min(), xs.max()
                cropped_arr = image_source[ymin:ymax+1, xmin:xmax+1, :]
                cropped_pil = Image.fromarray(cropped_arr)
            masked_pil_list.append(cropped_pil)

        pred_adapter = relabeler.relabel_regions(masked_pil_list, confidence_threshold=0.05)
        # "class_name (conf)" の表示用ラベルを一旦保持
        adapter_labels = [
            f"{p['class_name']} ({p['confidence']:.2f})" for p in pred_adapter
        ]
        # ここで WBFスコアに使う confidence は「アダプタの confidence」とする
        adapter_scores = np.array([p['confidence'] for p in pred_adapter], dtype=float)

        # ------------------------------------------------------
        # text_prompt_new 側は GroundingDINO 由来の phrases_new をラベルとして使う
        # ------------------------------------------------------
        # scores_new: GroundingDINO の confidence
        # no_adapter_labels: "phrase (conf)"
        # （WBFには scores_new をそのまま使用）
        # ------------------------------------------------------

        # 6) WBF 用に2つの結果をまとめる
        #    text_prompt(アダプタ) 側: (input_boxes_prompt, adapter_scores, adapter_labels)
        #    text_prompt_new(GroundingDINOフレーズ) : (input_boxes_new, scores_new, no_adapter_labels)
        all_bboxes = np.concatenate([input_boxes_prompt, input_boxes_new], axis=0)
        all_scores = np.concatenate([adapter_scores, scores_new], axis=0)
        all_labels = adapter_labels + no_adapter_labels  # リスト結合

        fused_boxes, fused_confs, fused_labels = wbf(
            bboxes=all_bboxes.tolist(),
            scores=all_scores.tolist(),
            labels=all_labels,
            iou_threshold=0.5,  # 適宜
            n=1
        )

        fused_boxes = np.array(fused_boxes)
        fused_scores = np.array(fused_confs)  # WBF後のスコア
        # fused_labels は WBFで1つにまとめられた代表ラベルを格納

        # 7) 最終的な BBox に対して SAM2 でマスクを再度推定
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
            final_masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=fused_boxes,
                multimask_output=False
            )
        if final_masks.ndim == 4:
            final_masks = final_masks.squeeze(1)

        # 8) 可視化と JSON 出力
        # 表示用ラベル例: "CLIP: {fused_label}"
        final_vis_labels = [
            f"{lb}" for lb in fused_labels
        ]

        ann_fused = visualizer.visualize(image_source, fused_boxes, final_masks, final_vis_labels)

        out_img_path = os.path.join(args.output_dir, f"result_fused_{adapter_name}.jpg")
        cv2.imwrite(out_img_path, cv2.cvtColor(ann_fused, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved fused result to {out_img_path}")

        if args.dump_json:
            results = {
                "adapter_name": adapter_name,
                "image_path": args.input_image,
                "image_size": {"width": w, "height": h},
                "predictions_WBF": []
            }
            for i, (box, mk, sc, lb) in enumerate(zip(fused_boxes, final_masks, fused_scores, fused_labels)):
                rle = mask2rle(mk)
                results["predictions_WBF"].append({
                    "region_id": i,
                    "class_name": lb,    # 代表ラベル
                    "confidence": float(sc),
                    "bbox": box.tolist(),
                    "mask_rle": rle
                })
            json_out = os.path.join(args.output_dir, f"result_fused_{adapter_name}.json")
            with open(json_out, "w") as f:
                json.dump(results, f, indent=4)
            print(f"[INFO] Saved JSON to: {json_out}")

    print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser("SAM2 + 4 Adapters Inference (with WBF)", add_help=True)
    # --- 必須: SAM2, GroundingDINO ---
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--sam2_config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--grounded_config", type=str, required=True)

    # --- 4種類のアダプタ(学習済み) のチェックポイントフォルダ と epoch番号
    parser.add_argument("--clip_adapter_dir", type=str, required=True,
                        help="Path to CLIP-Adapter directory (e.g. checkpoints_ClipA/adapter)")
    parser.add_argument("--clip_adapter_epoch", type=int, default=10,
                        help="Which epoch tar to load?")

    parser.add_argument("--tip_adapter_dir", type=str, required=True,
                        help="Path to Tip-Adapter directory (e.g. checkpoints_TipA/adapter)")
    parser.add_argument("--tip_adapter_epoch", type=int, default=1)

    parser.add_argument("--tip_adapter_f_dir", type=str, required=True,
                        help="Path to Tip-Adapter-F directory (e.g. checkpoints_TipA/adapter_f/)")
    parser.add_argument("--tip_adapter_f_epoch", type=int, default=1)

    parser.add_argument("--cross_modal_dir", type=str, required=True,
                        help="Path to Cross-Modal directory (e.g. checkpoints_CrossModal/adapter)")
    parser.add_argument("--cross_modal_epoch", type=int, default=10)

    # --- 入力画像など ---
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)       # 従来
    parser.add_argument("--text_prompt_new", type=str, required=True)   # 新規追加
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="outputs_inference")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dump_json", action="store_true")
    args = parser.parse_args()

    process_image(args)


if __name__ == "__main__":
    main()
