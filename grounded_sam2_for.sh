#!/bin/bash

# Settings
INPUT_DIR="/mnt/source/cityscapes/train/"
OUTPUT_BASE_DIR="/mnt/media/SSD-PUTA/output_SAM2-CLIP/TOTAL_CLIP_ViT-B32_finetuned_2"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT="./checkpoints/sam2.1_hiera_large.pt"
GROUNDING_DINO_CONFIG="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT="gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD=0.16
TEXT_THRESHOLD=0.17
CLASS_NAMES="/mnt/source/Downloads/class_name.txt"
#TEXT_PROMPT="car. person. left arrow. straight arrow. right arrow. bicycle. truck. bus. traffic light. traffic sign. sidewalk. vegetation. building. guard rail. fence."
#TEXT_PROMPT="left arrow. straight arrow. right arrow. ego vehicle. sidewalk. parking. rail track. building. wall. fence. guard rail. bridge. tunnel. pole. polegroup. traffic light. traffic sign. vegetation. terrain. sky. person. rider. car. truck. bus. caravan. trailer. train. motorcycle. bicycle. license plate. Speed Limit 20 km/h. Speed Limit 30 km/h. Speed Limit 50 km/h. Speed Limit 60 km/h. Speed Limit 70 km/h. Speed Limit 80 km/h. Speed Limit 80 km/h end. Speed Limit 100 km/h. Speed Limit 120 km/h. No passing. No passing for trucks. Right of way at next intersection. Main road. Yield. Stop. No vehicles. No trucks. No entry. General caution. Dangerous curve left. Dangerous curve right. Winding road. Bumpy road. Slippery road. Road narrows on the right. Road work. Traffic lights. Pedestrians. Children crossing. Bike crossing. Beware of ice/snow. Wild animals crossing. End of all speed and passing limits. Turn right. Turn left. Only straight. Only straight or right. Only straight or left. Keep right. Keep left. Roundabout mandatory. End of overtaking limit. End of overtaking limit for trucks."
#TEXT_PROMPT="arrow. road marker. lane. ego vehicle. sidewalk. parking. rail track. building. wall. fence. guard rail. bridge. tunnel. pole. polegroup. traffic light. traffic sign. vegetation. terrain. sky. person. rider. car. truck. bus. caravan. trailer. train. motorcycle. bicycle. license plate."
TEXT_PROMPT="traffic sign. arrow. right. left. straight. road marker"
DEVICE="cuda"

# Process all PNG files in the input directory
for INPUT_IMAGE in "$INPUT_DIR"*.png; do
    # Create output directory based on the file name
    FILE_NAME=$(basename "$INPUT_IMAGE")
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${FILE_NAME%.*}"
    mkdir -p "$OUTPUT_DIR"
    
    # Build and execute the command
    python grounded_sam2_demo_CLIP_20241203v5.py \
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
        --clip_checkpoint "/mnt/source/Downloads/best_model_total_clip_200.pt" \
        --dump_json
done
