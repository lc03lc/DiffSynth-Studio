CATEGORIES=(
  cloth
  grab_cube
  grab_cup
  grab_pencil
  grab_to_plate
  pull_plate
  push_box
  push_plate
  roll
  writing
)

category=${CATEGORIES[1]}
prev_category=${CATEGORIES[0]}

# 选卡
export CUDA_VISIBLE_DEVICES=7

# 仅挂载，不训练
export DIT_LORA_PATH="/raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_dit_lora/${category}/epoch-1.safetensors"
export DIT_LORA_ALPHA="1.0"

DATA_ROOT="/raid/workspace1/lc/HumanAndRobot/Datasets/HumanAndRobot_split_category/${category}"
BASE="/raid/share_model/Wan2.1-VACE-1.3B"

accelerate launch --num_processes 1 --mixed_precision bf16 \
examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATA_ROOT/" \
  --dataset_metadata_path "$DATA_ROOT/metadata_2.csv" \
  --data_file_keys "video,vace_video" \
  --height 480 --width 832 --num_frames 81 \
  --dataset_repeat 10 \
  --model_paths "[
    [
      \"$BASE/diffusion_pytorch_model.safetensors\"
    ],
    \"$BASE/models_t5_umt5-xxl-enc-bf16.pth\",
    \"$BASE/Wan2.1_VAE.pth\"
  ]" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "/raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_vace_lora/${category}" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64 \
  --extra_inputs "vace_video" \
  --use_gradient_checkpointing_offload \
  --lora_checkpoint "/raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_vace_lora/${prev_category}/epoch-1.safetensors"
