#!/bin/bash

# Settings
INPUT_DIR="/mnt/source/cityscapes/train/"
OUTPUT_BASE_DIR="/mnt/media/SSD-PUTA/output_SAM2-CLIP/TOTAL_CLIP_ViT-B32_finetuned_v12"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT="./checkpoints/sam2.1_hiera_large.pt"
GROUNDING_DINO_CONFIG="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT="gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD=0.16
TEXT_THRESHOLD=0.17

# Tip-Adapter-F 関連ファイル (キャッシュ + fine-tuned weight)
TIP_ADAPTER_CACHE="/mnt/source/Downloads/tip_adapter_F_cache.pt"
ADAPTER_WEIGHT="/mnt/source/Downloads/best_F_16shots.pt"  # 例: fine-tuning後のweight

# テキストプロンプト (GroundingDINO用)
TEXT_PROMPT="traffic sign. arrow. right. left. straight. road marker"

DEVICE="cuda"

# Process all PNG files in the input directory
for INPUT_IMAGE in "$INPUT_DIR"*.png; do
    # Create output directory based on the file name
    FILE_NAME=$(basename "$INPUT_IMAGE")
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${FILE_NAME%.*}"
    mkdir -p "$OUTPUT_DIR"
    
    # Build and execute the command
    python grounded_sam2_demo_CLIP_20241231v12.py \
        --sam2_checkpoint "$SAM2_CHECKPOINT" \
        --sam2_config "$SAM2_CONFIG" \
        --grounded_checkpoint "$GROUNDING_DINO_CHECKPOINT" \
        --grounded_config "$GROUNDING_DINO_CONFIG" \
        --input_image "$INPUT_IMAGE" \
        --text_prompt "$TEXT_PROMPT" \
        --output_dir "$OUTPUT_DIR" \
        --box_threshold "$BOX_THRESHOLD" \
        --text_threshold "$TEXT_THRESHOLD" \
        --device "$DEVICE" \
        --clip_checkpoint "$TIP_ADAPTER_CACHE" \
        --adapter_weight "$ADAPTER_WEIGHT" \
        --dump_json
done
