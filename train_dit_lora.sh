#!/usr/bin/env bash
set -e

# ---- 路径与超参 ----
DATA_ROOT="/raid/workspace1/lc/HumanAndRobot/Datasets/HumanAndRobot_split_category/cloth"
META_CSV="$DATA_ROOT/metadata.csv"          # 包含: video,prompt
BASE="/raid/share_model/Wan2.1-VACE-1.3B"
OUT="/raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_dit_lora/cloth"
HEIGHT=480
WIDTH=832
FRAMES=81
LR=1e-4
EPOCHS=5
RANK=64

# ---- 启动训练（DiT LoRA）----
CUDA_VISIBLE_DEVICES=4 \
accelerate launch --num_processes 1 examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_metadata_path "$META_CSV" \
  --height $HEIGHT --width $WIDTH --num_frames $FRAMES \
  --dataset_repeat 10 \
  --model_paths "[
    [
      \"$BASE/diffusion_pytorch_model.safetensors\"
    ],
    \"$BASE/models_t5_umt5-xxl-enc-bf16.pth\",
    \"$BASE/Wan2.1_VAE.pth\"
  ]" \
  --learning_rate $LR \
  --num_epochs $EPOCHS \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank $RANK \
  --use_gradient_checkpointing_offload
