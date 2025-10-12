#!/usr/bin/env bash
set -euo pipefail

# ==== 基本配置 ====
REPO_DIR="/raid/workspace1/lc/HumanAndRobot/DiffSynth-Studio"
DATASET_ROOT="/raid/workspace1/lc/HumanAndRobot/Datasets/HumanAndRobot_split_category"
OUTPUT_ROOT="/raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_dit_lora"
LOGDIR="/raid/workspace1/lc/HumanAndRobot/Logs/wan_lora_dit_gpu4"
BASE="/raid/share_model/Wan2.1-VACE-1.3B"   # 1.3B
mkdir -p "$OUTPUT_ROOT" "$LOGDIR"
cd "$REPO_DIR"

# 训练超参
HEIGHT=480; WIDTH=832; FRAMES=81
LR=1e-4; EPOCHS=2; RANK=64
DATASET_REPEAT=10
MIXED_PRECISION="bf16"   # 不支持就改为 fp16

MODEL_PATHS_JSON="[ [\"$BASE/diffusion_pytorch_model.safetensors\"], \"$BASE/models_t5_umt5-xxl-enc-bf16.pth\", \"$BASE/Wan2.1_VAE.pth\" ]"

# 该卡要跑的类别
CATS=(cloth grab_cube grab_cup grab_pencil)

export CUDA_VISIBLE_DEVICES=4
for category in "${CATS[@]}"; do
  DATA_ROOT="${DATASET_ROOT}/${category}"
  META_CSV="${DATA_ROOT}/metadata.csv"
  OUT="${OUTPUT_ROOT}/${category}"
  mkdir -p "$OUT"

  if [[ ! -f "$META_CSV" ]]; then
    echo "[GPU4][SKIP] ${category}: ${META_CSV} 不存在" | tee -a "$LOGDIR/${category}.log"
    continue
  fi

  echo "[GPU4][RUN] ${category} -> ${OUT}"
  accelerate launch --num_processes 1 --mixed_precision "$MIXED_PRECISION" \
    examples/wanvideo/model_training/train.py \
      --dataset_base_path "$DATA_ROOT" \
      --dataset_metadata_path "$META_CSV" \
      --height "$HEIGHT" --width "$WIDTH" --num_frames "$FRAMES" \
      --dataset_repeat "$DATASET_REPEAT" \
      --model_paths "$MODEL_PATHS_JSON" \
      --learning_rate "$LR" \
      --num_epochs "$EPOCHS" \
      --remove_prefix_in_ckpt "pipe.dit." \
      --output_path "$OUT" \
      --lora_base_model "dit" \
      --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
      --lora_rank "$RANK" \
      --use_gradient_checkpointing_offload \
      2>&1 | tee "$LOGDIR/${category}.log"

  echo "[GPU4][DONE] ${category}"
done

echo "[GPU4] 全部完成 ✅"
