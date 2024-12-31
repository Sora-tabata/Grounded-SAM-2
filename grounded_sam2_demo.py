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

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser("Grounded SAM2 Demo", add_help=True)
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build SAM2 model
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    # Build Grounding DINO model
    grounding_model = load_model(
        model_config_path=args.grounded_config,
        model_checkpoint_path=args.grounded_checkpoint,
        device=args.device
    )
    
    # Load image and set up SAM2 predictor
    image_source, image = load_image(args.input_image)
    sam2_predictor.set_image(image_source)
    
    # Run Grounding DINO model
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=args.text_prompt.lower().strip() + ".",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device
    )
    
    # Process boxes for SAM2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # Perform SAM2 prediction
    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            # Enable TF32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    
    # Post-process the output
    # Convert masks shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    confidences = confidences.numpy().tolist()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))
    
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    
    # Visualization
    img = cv2.imread(args.input_image)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )
    
    # Annotate and save images
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    cv2.imwrite(os.path.join(args.output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame_with_mask = mask_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    cv2.imwrite(os.path.join(args.output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame_with_mask)
    
    # Save results as JSON
    if args.dump_json:
        mask_rles = [single_mask_to_rle(mask) for mask in masks]
        input_boxes_list = input_boxes.tolist()
        scores_list = scores.tolist()
        
        results = {
            "image_path": args.input_image,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes_list, mask_rles, scores_list)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        with open(os.path.join(args.output_dir, "grounded_sam2_results.json"), "w") as f:
            json.dump(results, f, indent=4)

