export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INTRA_MODEL_NAME="yang1232009/DC-ControlNet"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export NCCL_P2P_DISABLE=1

accelerate launch --config_file configs/zero2.yaml --main_process_port 29500 --num_processes 4 train_inter_element_controller.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_intra_model_name_or_path=$INTRA_MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_folder ./dmc-120k-multicondition-2objs \
  --regularization_annotation ./generated_data_with_flux_2objs_final_subjects_split_v3.json \
  --image_column image \
  --caption_column caption \
  --conditioning_image_column image \
  --resolution=1024 \
  --prompt_dropout=0.0 \
  --dataloader_num_workers 4 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=80000 \
  --learning_rate=5e-5 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --validation_multi \
  --validation_image_list \
    "./assets/condition/condition-1/zoe/11.png|./assets/condition/condition-1/canny/32.png|./assets/background_images/background_garden_content.png" \
    "./assets/condition/condition-1/canny/32.png|./assets/condition/condition-1/zoe/11.png|./assets/background_images/background_garden_content.png" \
    "./assets/condition/condition-1/canny/bag.png|./assets/condition/condition-1/canny/119198.png|./assets/background_images/background_living_room_content.png" \
    "./assets/condition/condition-1/canny/119198.png|./assets/condition/condition-1/canny/bag.png|./assets/background_images/background_living_room_content.png" \
  --validation_layout_image_list \
    "./assets/layout/layout-1/dot/11.png|./assets/layout/layout-1/box/bear_box.png|./assets/background_images/background_layout.png" \
    "./assets/layout/layout-1/box/bear_box.png|./assets/layout/layout-1/dot/11.png|./assets/background_images/background_layout.png" \
    "./assets/layout/layout-1/box/bag_box.png|./assets/layout/layout-1/dot/bear_box.png|./assets/background_images/background_layout.png" \
    "./assets/layout/layout-1/dot/bear_box.png|./assets/layout/layout-1/box/bag_box.png|./assets/background_images/background_layout.png" \
  --validation_control_type_list "2|0|3" "0|2|3" "0|0|3" "0|0|3" \
  --validation_layout_type_list "0|1|1" "1|0|1" "1|1|1" "1|1|1" \
  --validation_condition_prompt_list \
    "teddy bear|cat|sunny garden" "cat|teddy bear|sunny garden" "a bag|a bear|cozy living room" "a bear|a bag|cozy living room" \
  --validation_prompt "A teddy bear and a cat in a sunny garden." "A teddy bear and a cat in a sunny garden." "A bag and a bear in a cozy living room." "A bag and a bear in a cozy living room." \
  --validation_steps=500 \
  --max_train_samples=100 \
  --checkpointing_steps=10000 \
  --report_to="wandb" \
  --output_dir="sdxl-inter_element_controller"
