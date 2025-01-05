#!/bin/bash

# ------------------------------------------------------
# Settings
# ------------------------------------------------------
INPUT_DIR="/mnt/source/cityscapes/train/"
OUTPUT_BASE_DIR="/mnt/media/SSD-PUTA/output_SAM2-CLIP/TOTAL_CLIP_ViT-B32_finetuned_v13"

SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT="./checkpoints/sam2.1_hiera_large.pt"

GROUNDING_DINO_CONFIG="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT="gdino_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD=0.16
TEXT_THRESHOLD=0.17
TEXT_PROMPT="traffic sign. arrow. right. left. straight. road marker"
DEVICE="cuda"

# --- 4種類のアダプタに対応したチェックポイント場所 & epoch番号を指定 ---
#   例: checkpoints_ClipA/adapter/model.pth.tar-10
CLIP_ADAPTER_DIR="/mnt/source/Grounded-SAM-2/checkpoints/checkpoints_ClipA/adapter"
CLIP_ADAPTER_EPOCH=10

TIP_ADAPTER_DIR="/mnt/source/Grounded-SAM-2/checkpoints/checkpoints_TipA/adapter"
TIP_ADAPTER_EPOCH=1

TIP_ADAPTER_F_DIR="/mnt/source/Grounded-SAM-2/checkpoints/checkpoints_TipA/adapter_tipa-f-"
TIP_ADAPTER_F_EPOCH=1

CROSS_MODAL_DIR="/mnt/source/Grounded-SAM-2/checkpoints/checkpoints_CrossModal/adapter"
CROSS_MODAL_EPOCH=10

# ------------------------------------------------------
# Process all PNG files in the input directory
# ------------------------------------------------------
for INPUT_IMAGE in "$INPUT_DIR"*.png; do
    # Create output directory based on the file name
    FILE_NAME=$(basename "$INPUT_IMAGE")
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${FILE_NAME%.*}"
    mkdir -p "$OUTPUT_DIR"
    
    # Build and execute the command
    python grounded_sam2_demo_CLIP_20241231v13.py \
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
        --dump_json \
        \
        --clip_adapter_dir "$CLIP_ADAPTER_DIR" \
        --clip_adapter_epoch "$CLIP_ADAPTER_EPOCH" \
        --tip_adapter_dir "$TIP_ADAPTER_DIR" \
        --tip_adapter_epoch "$TIP_ADAPTER_EPOCH" \
        --tip_adapter_f_dir "$TIP_ADAPTER_F_DIR" \
        --tip_adapter_f_epoch "$TIP_ADAPTER_F_EPOCH" \
        --cross_modal_dir "$CROSS_MODAL_DIR" \
        --cross_modal_epoch "$CROSS_MODAL_EPOCH"

done
