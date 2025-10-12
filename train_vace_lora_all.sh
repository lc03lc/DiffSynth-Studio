#!/usr/bin/env bash
set -euo pipefail

# 类别列表
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

# 选卡
export CUDA_VISIBLE_DEVICES=7

# 固定路径
BASE="/raid/share_model/Wan2.1-VACE-1.3B"
ROOT_OUT_DIT="/raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_dit_lora"
ROOT_OUT_VACE="/raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_vace_lora"
DATA_ROOT_BASE="/raid/workspace1/lc/HumanAndRobot/Datasets/HumanAndRobot_split_category"

# LoRA 仅挂载，不训练
export DIT_LORA_ALPHA="1.0"

# 依次从 (1,0) 到 (9,8)
for i in $(seq 1 9); do
  category="${CATEGORIES[$i]}"
  prev_index=$((i-1))
  prev_category="${CATEGORIES[$prev_index]}"

  echo "=============================="
  echo "Start: category=${category}, prev_category=${prev_category}"
  echo "=============================="

  # 路径设置
  export DIT_LORA_PATH="${ROOT_OUT_DIT}/${category}/epoch-1.safetensors"
  DATA_ROOT="${DATA_ROOT_BASE}/${category}"
  OUTPUT_PATH="${ROOT_OUT_VACE}/${category}"
  LORA_CKPT="${ROOT_OUT_VACE}/${prev_category}/epoch-1.safetensors"

  # 可选：若上一个类别的 vace LoRA 不存在就跳过（需要严格执行可移除该判断）
  if [[ ! -f "${LORA_CKPT}" ]]; then
    echo "WARN: 找不到上一个类别的LoRA权重：${LORA_CKPT}，跳过该轮。"
    continue
  fi

  accelerate launch --num_processes 1 --mixed_precision bf16 \
  examples/wanvideo/model_training/train.py \
    --dataset_base_path "${DATA_ROOT}/" \
    --dataset_metadata_path "${DATA_ROOT}/metadata_2.csv" \
    --data_file_keys "video,vace_video" \
    --height 480 --width 832 --num_frames 81 \
    --dataset_repeat 10 \
    --model_paths "[
      [
        \"${BASE}/diffusion_pytorch_model.safetensors\"
      ],
      \"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",
      \"${BASE}/Wan2.1_VAE.pth\"
    ]" \
    --learning_rate 1e-4 \
    --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.vace." \
    --output_path "${OUTPUT_PATH}" \
    --lora_base_model "vace" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 64 \
    --extra_inputs "vace_video" \
    --use_gradient_checkpointing_offload \
    --lora_checkpoint "${LORA_CKPT}"

  echo "Done: category=${category}, prev_category=${prev_category}"
done

echo "全部任务完成。"
