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

# ===== ここがポイント: adapters.py の中身を直接 or 関数として利用する =====
# 今回は同じディレクトリに "adapters.py" がある前提でimport (パスは適宜修正してください)
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


# ----------------------------------------------------------------------------
# 1) 学習済みアダプタ (tar-10, tar-1など) を推論用にロードするためのラッパ
# ----------------------------------------------------------------------------
class InferenceAdapterTrainer(ADAPTER):
    """
    adapters.py の ADAPTER を継承し、推論に必要な部分だけ初期化＆利用するための簡易クラス。
    """
    def __init__(self, adapter_init_mode, device='cuda'):
        # Dasslのcfgを最低限用意
        # train_v12_total.py でも使っている get_cfg_default() を使う
        # extend_cfg_for_adapter呼ぶとすでにfreezeされる

        # → ここでは defrostしてから書き換える
        self.cfg = get_cfg_default()
        extend_cfg_for_adapter(self.cfg)
        self.cfg.defrost()
        # 好みで細かい設定
        self.cfg.MODEL.BACKBONE.NAME = "ViT-B/32"
        self.cfg.TRAINER.ADAPTER.INIT = adapter_init_mode
        self.cfg.TRAINER.ADAPTER.PREC = "fp32"  # amp/fp16にするなら調整
        self.cfg.TRAINER.NAME = "ADAPTER"
        self.cfg.DATASET.NAME = "CIFAR10"  # ダミー
        self.cfg.CLASSNAMES = selected_labels
        # 出力先はダミー
        self.cfg.OUTPUT_DIR = "./checkpoints_infer_tmp"
        self.cfg.freeze()
        # 2) ★ lab2cname を先に作っておく
        self.classnames = selected_labels
        self.lab2cname = {i: cname for i, cname in enumerate(self.classnames)}

        # ADAPTERクラスの親コンストラクタを呼び出す
        super().__init__(self.cfg)

        # クラス名セット
        self.classnames = selected_labels
        self.device = torch.device(device)
        self.start_epoch = 0
        self.max_epoch = 1  # 学習はしないので1に
    # def build_evaluator(self, **kwargs):
    #     """
    #     評価器を作らずに None を返すようにする。
    #     すると lab2cname へのアクセスも回避できる。
    #     """
    #     return None
    def build_data_loader(self):
        """
        推論だけなら DataLoader は不要。
        """
        pass

    def train(self):
        """
        推論だけなので学習はスキップ
        """
        pass
    
    def build_model(self):
        """
        親クラスで build_model() が呼ばれるのを抑制するために、空にしておく。
        """
        pass

    # def build_model(self):
    #     """
    #     adapters.py の ADAPTER.build_model() とほぼ同じ実装。
    #     ここでは「CustomCLIP」インスタンスを self.model に持つ。
    #     """
    #     cfg = self.cfg
    #     # CLIP をロード
    #     clip_model = load_clip_to_cpu(cfg)
    #     if torch.cuda.is_available():
    #         clip_model = clip_model.cuda()
    #     clip_model.float()
    #     self.classnames = selected_labels
    #     self.lab2cname = {i: cname for i, cname in enumerate(self.classnames)}

    #     # CustomCLIP を構築
    #     self.model = CustomCLIP(cfg, self.classnames, clip_model)

    #     # adapter 以外の勾配を切る (推論なので無視でもOK)
    #     for name, param in self.model.named_parameters():
    #         if "adapter" not in name:
    #             param.requires_grad_(False)

    #     self.model.to(self.device)
    #     self.model = self.model.float()

    #     # Optimizer は推論には不要だがエラー回避のためダミー設定
    #     self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
    #     self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
    #     self.register_model("adapter", self.model.adapter, self.optim, self.sched)

    def load_model_from_dir(self, ckpt_dir, epoch=None):
        """
        adapters.py の load_model() を呼び出して tar-10 等をロードする。
        例: epoch=10 なら "model.pth.tar-10" が読み込まれる
        """
        print(f"[INFO] Load model from dir: {ckpt_dir}, epoch={epoch}")
        super().load_model(ckpt_dir, self.cfg, epoch=epoch)  # 親の load_model() を使う
        # 読み込まれたモデルは self.model (CustomCLIP) に格納済み
        
        
    def build_inference_model(self):
        """
        実際のビルド処理をこちらで行う (元々 build_model で書いていた内容)。
        """
        cfg = self.cfg
        clip_model = load_clip_to_cpu(cfg)
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
        clip_model.float()

        self.classnames = selected_labels
        self.lab2cname = {i: cname for i, cname in enumerate(self.classnames)}

        # 例: build_inference_model の中などで
        clip_model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.preprocess_fn = preprocess  # ここでtransform関数を保持
        self.clip_model = clip_model
        self.model = CustomCLIP(cfg, self.classnames, clip_model)
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)
        self.model.to(self.device)
        self.model = self.model.float()

        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # 重要: このタイミングで register_model("adapter", ...) を呼ぶ
        self.register_model("adapter_inference", self.model.adapter, self.optim, self.sched)
        # ↑ 登録名を "adapter_inference" など別名にしておくと重複のリスクがさらに下がる


    @torch.no_grad()
    def infer_region(self, region_tensor: torch.Tensor) -> torch.Tensor:
        """
        引数: region_tensor.shape = (1, 3, H, W)
        返値: logits.shape = (1, num_classes)
        """
        self.set_model_mode("eval")
        # features 推論
        logits = self.model.forward_features(region_tensor.to(self.device))
        return logits


# ----------------------------------------------------------------------------
# 2) 上記ラッパを使って、CLIP-Adapter / Tip-Adapter / Tip-Adapter-F / CrossModal を
#    それぞれロード・推論するクラス
# ----------------------------------------------------------------------------
class AdapterRelabeler:
    def __init__(self, adapter_init_mode: str, ckpt_dir: str, epoch_num: int, device='cuda'):
        ...
        self.trainer = InferenceAdapterTrainer(adapter_init_mode, device=device)
        self.trainer.build_inference_model()
        self.trainer.load_model_from_dir(ckpt_dir, epoch=epoch_num)
        self.class_names = selected_labels
        self.device = torch.device(device)
        
    @torch.no_grad()
    def _encode_image(self, pil_image: Image.Image):
        """領域画像を CLIP encode して L2 正規化"""
        img_tensor = self.trainer.preprocess_fn(pil_image).unsqueeze(0).to(self.device)
        feat = self.clip_model.encode_image(img_tensor).float()
        feat = F.normalize(feat, p=2, dim=-1)
        return feat  # shape: (1, dim)

    @torch.no_grad()
    def relabel_regions(self, pil_images, confidence_threshold=0.02):
        """
        pil_images: [N個の領域画像 (PIL)] 
        """
        predictions = []

        for pil_img in pil_images:
            # 1) PIL → Tensor化（CLIPのpreprocess）
            img_tensor = self.trainer.preprocess_fn(pil_img).unsqueeze(0).to(self.device)
            
            # 2) CLIPで埋め込みを計算する場合 (シンプルに直接 encode_image したい場合)
            #    ※ adapters.pyのロジックは self.trainer.infer_region() 経由で行われるが、
            #      ここでは分かりやすく encode_image → normalize の手順を明示します。

            #    self.trainer.model は CustomCLIP のインスタンス
            #    通常は self.trainer.model.forward_features(...) で推論しますが、
            #    直接 encode_image を呼ぶ例を示します:
            clip_features = self.trainer.model.clip_model.encode_image(img_tensor)
            clip_features = F.normalize(clip_features, p=2, dim=-1)  # 正規化

            # 3) アダプタも含めた最終 logits 推定をするなら:
            #    - self.trainer.infer_region() を呼ぶ (adapters.py のロジックを使う) か
            #    - ここで直接 self.trainer.model.forward_features(...) を呼ぶ
            logits = self.trainer.model.forward_features(clip_features)

            # 4) ソフトマックスでクラス確率を得る
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            # 5) 最高スコアのクラスを判定
            top_idx = np.argmax(probs)
            top_class = self.class_names[top_idx]
            confidence = float(probs[top_idx])

            if confidence < confidence_threshold:
                top_class = "unknown"

            # top-k も格納
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

        return predictions



# ----------------------------------------------------------------------------
# 可視化やマスク処理は既存のロジックと同様 (例)
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
# メイン処理: SAM2 でセグメンテーション → 領域を上記のアダプタ4種でラベリング
# ----------------------------------------------------------------------------
def process_image(args):
    device = torch.device(args.device)

    # 1) SAM2, GroundingDINO のロード
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=device
    )

    # 2) 各アダプタをロード (tar-* ファイルから)
    clip_adapter_relabeler = AdapterRelabeler(
        adapter_init_mode="ClipA",
        ckpt_dir=args.clip_adapter_dir,   # e.g. "./checkpoints_ClipA/adapter"
        epoch_num=args.clip_adapter_epoch,  # e.g. 10
        device=device
    )
    tip_adapter_relabeler = AdapterRelabeler(
        adapter_init_mode="TipA",
        ckpt_dir=args.tip_adapter_dir,    # e.g. "./checkpoints_TipA/adapter"
        epoch_num=args.tip_adapter_epoch,   # e.g. 1
        device=device
    )
    tip_adapter_f_relabeler = AdapterRelabeler(
        adapter_init_mode="TipA-f-",
        ckpt_dir=args.tip_adapter_f_dir,  # e.g. "./checkpoints_TipA/adapter_tipa-f-"
        epoch_num=args.tip_adapter_f_epoch,  # e.g. 1
        device=device
    )
    cross_modal_relabeler = AdapterRelabeler(
        adapter_init_mode="CrossModal",
        ckpt_dir=args.cross_modal_dir,    # e.g. "./checkpoints_CrossModal/adapter"
        epoch_num=args.cross_modal_epoch,   # e.g. 10
        device=device
    )

    visualizer = ResultVisualizer()

    # 3) 入力画像ロード
    os.makedirs(args.output_dir, exist_ok=True)

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

    # 6) マスク領域を PIL でクロップ & 各アダプタで推論
    #    ここでは簡易的に (ymin:ymax, xmin:xmax) でクロップした PIL をまとめて relabel
    #    実装詳細はお好みで
    masked_pil_list = []
    for mk in masks:
        mk_bool = mk.astype(bool)
        if not mk_bool.any():
            # マスクが真っ白 or 真っ黒などの場合、全体を返す or 適宜ハンドリング
            cropped_pil = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR))  
        else:
            arr = np.array(image_source)
            ys, xs = np.where(mk_bool)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            cropped_arr = arr[ymin:ymax+1, xmin:xmax+1, :]
            cropped_pil = Image.fromarray(cropped_arr)
        masked_pil_list.append(cropped_pil)

    # (A) CLIP-Adapter
    pred_clip = clip_adapter_relabeler.relabel_regions(masked_pil_list, confidence_threshold=0.05)
    labels_clip = [f"{p['class_name']} ({p['confidence']:.2f})" for p in pred_clip]
    ann_clip = visualizer.visualize(image_source, input_boxes, masks, labels_clip)
    out_clip = os.path.join(args.output_dir, "result_clip_adapter.jpg")
    cv2.imwrite(out_clip, cv2.cvtColor(ann_clip, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved CLIP-Adapter result to {out_clip}")

    # (B) Tip-Adapter
    pred_tip = tip_adapter_relabeler.relabel_regions(masked_pil_list, confidence_threshold=0.05)
    labels_tip = [f"{p['class_name']} ({p['confidence']:.2f})" for p in pred_tip]
    ann_tip = visualizer.visualize(image_source, input_boxes, masks, labels_tip)
    out_tip = os.path.join(args.output_dir, "result_tip_adapter.jpg")
    cv2.imwrite(out_tip, cv2.cvtColor(ann_tip, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved Tip-Adapter result to {out_tip}")

    # (C) Tip-Adapter-F
    pred_tipf = tip_adapter_f_relabeler.relabel_regions(masked_pil_list, confidence_threshold=0.05)
    labels_tipf = [f"{p['class_name']} ({p['confidence']:.2f})" for p in pred_tipf]
    ann_tipf = visualizer.visualize(image_source, input_boxes, masks, labels_tipf)
    out_tipf = os.path.join(args.output_dir, "result_tip_adapter_f.jpg")
    cv2.imwrite(out_tipf, cv2.cvtColor(ann_tipf, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved Tip-Adapter-F result to {out_tipf}")

    # (D) Cross-Modal
    pred_cross = cross_modal_relabeler.relabel_regions(masked_pil_list, confidence_threshold=0.05)
    labels_cross = [f"{p['class_name']} ({p['confidence']:.2f})" for p in pred_cross]
    ann_cross = visualizer.visualize(image_source, input_boxes, masks, labels_cross)
    out_cross = os.path.join(args.output_dir, "result_cross_modal.jpg")
    cv2.imwrite(out_cross, cv2.cvtColor(ann_cross, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved Cross-Modal result to {out_cross}")

    # 7) JSON 出力 (オプション)
    
    # (A) CLIP-Adapter
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "image_size": {"width": w, "height": h},
            "predictions_CLIPAdapter": []
        }
        for i, (pred, box, mk) in enumerate(zip(pred_clip, input_boxes, masks)):
            rle = mask2rle(mk)
            results["predictions_CLIPAdapter"].append({
                "region_id": i,
                "class_name": pred["class_name"],
                "confidence": pred["confidence"],
                "bbox": box.tolist(),
                "mask_rle": rle
            })
        json_out = os.path.join(args.output_dir, "result_clip_adapter.json")
        with open(json_out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Saved JSON to: {json_out}")
        
    
    # (B) Tip-Adapter
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "image_size": {"width": w, "height": h},
            "predictions_TipAdapter": []
        }
        for i, (pred, box, mk) in enumerate(zip(pred_tip, input_boxes, masks)):
            rle = mask2rle(mk)
            results["predictions_TipAdapter"].append({
                "region_id": i,
                "class_name": pred["class_name"],
                "confidence": pred["confidence"],
                "bbox": box.tolist(),
                "mask_rle": rle
            })
        json_out = os.path.join(args.output_dir, "result_tip_adapter.json")
        with open(json_out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Saved JSON to: {json_out}")
    
    # (C) Tip-Adapter-F
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "image_size": {"width": w, "height": h},
            "predictions_TipAdapterF": []
        }
        for i, (pred, box, mk) in enumerate(zip(pred_tipf, input_boxes, masks)):
            rle = mask2rle(mk)
            results["predictions_TipAdapterF"].append({
                "region_id": i,
                "class_name": pred["class_name"],
                "confidence": pred["confidence"],
                "bbox": box.tolist(),
                "mask_rle": rle
            })
        json_out = os.path.join(args.output_dir, "result_tip_adapter_f.json")
        with open(json_out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Saved JSON to: {json_out}")
    
    # (D) Cross-Modal
    if args.dump_json:
        results = {
            "image_path": args.input_image,
            "image_size": {"width": w, "height": h},
            "predictions_CrossAdapter": []
        }
        for i, (pred, box, mk) in enumerate(zip(pred_cross, input_boxes, masks)):
            rle = mask2rle(mk)
            results["predictions_CrossAdapter"].append({
                "region_id": i,
                "class_name": pred["class_name"],
                "confidence": pred["confidence"],
                "bbox": box.tolist(),
                "mask_rle": rle
            })
        json_out = os.path.join(args.output_dir, "result_cross_adapter.json")
        with open(json_out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Saved JSON to: {json_out}")
        


    print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser("SAM2 + tar-10 Adapters Inference", add_help=True)
    # --- 必須: SAM2, GroundingDINO ---
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--sam2_config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--grounded_config", type=str, required=True)

    # --- 4種類のアダプタ(学習済み) のチェックポイントフォルダ と epoch番号
    parser.add_argument("--clip_adapter_dir", type=str, required=True,
                        help="Path to CLIP-Adapter directory (e.g. checkpoints_ClipA/adapter)")
    parser.add_argument("--clip_adapter_epoch", type=int, default=10,
                        help="Which epoch tar to load? (e.g. 10 => model.pth.tar-10)")

    parser.add_argument("--tip_adapter_dir", type=str, required=True,
                        help="Path to Tip-Adapter directory (e.g. checkpoints_TipA/adapter)")
    parser.add_argument("--tip_adapter_epoch", type=int, default=1)

    parser.add_argument("--tip_adapter_f_dir", type=str, required=True,
                        help="Path to Tip-Adapter-F directory (e.g. checkpoints_TipA/adapter_tipa-f-)")
    parser.add_argument("--tip_adapter_f_epoch", type=int, default=1)

    parser.add_argument("--cross_modal_dir", type=str, required=True,
                        help="Path to Cross-Modal directory (e.g. checkpoints_CrossModal/adapter)")
    parser.add_argument("--cross_modal_epoch", type=int, default=10)

    # --- 入力画像など ---
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="outputs_inference")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dump_json", action="store_true")
    args = parser.parse_args()

    process_image(args)


if __name__ == "__main__":
    main()
