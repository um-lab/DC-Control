export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export NCCL_P2P_DISABLE=1

accelerate launch --config_file configs/zero2.yaml --main_process_port 29500 --num_processes 4 train_intra_element_controller.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_folder ./dmc120k \
  --regularization_annotation ./generated_data_1obj_final_named.json \
  --image_column image \
  --caption_column caption \
  --conditioning_image_column image \
  --resolution=1024 \
  --prompt_dropout=0.0 \
  --dataloader_num_workers 4 \
  --max_train_samples=95000 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=80000 \
  --learning_rate=5e-5 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --validation_image "./assets/condition/condition-1/zoe/11.png" "./assets/condition/condition-1/zoe/11.png" "./assets/condition/condition-1/canny/32.png" "./assets/condition/condition-1/canny/32.png" \
  --validation_layout_image "./assets/layout/layout-1/dot/11.png" "./assets/layout/layout-1/dot/32.png" "./assets/layout/layout-1/box/32.png" "./assets/layout/layout-1/box/3122.png" \
  --validation_condition_prompt "teddy bear" "teddy bear" "cat" "cat" \
  --validation_prompt "A teddy bear in a battlefield" "A teddy bear in a battlefield" "A cat in a science museum." "A cat in a science museum." \
  --validation_control_type 2 2 0 0 \
  --validation_layout_type 0 0 1 1 \
  --validation_steps=500 \
  --checkpointing_steps=10000 \
  --report_to="wandb" \
  --output_dir="sdxl-intra_element_controller"
