# V2V（编辑）：加载两份 LoRA，指定 vace_video
CUDA_VISIBLE_DEVICES=5 \
python infer_dit_plus_vace_lora.py \
  --base_dir /raid/share_model/Wan2.1-VACE-1.3B \
  --dit_lora  /raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_dit_lora/cloth/epoch-1.safetensors \
  --vace_lora /raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_vace_lora/grab_cup/epoch-1.safetensors \
  --prompt "The mechanical arm fold cloth" \
  --vace_video /raid/workspace1/lc/HumanAndRobot/Datasets/HumanAndRobot_split_category/cloth/human/001.mp4 \
  --width 832 --height 480 --num_frames 121 --fps 24 \
  --alpha_dit 1.0 --alpha_vace 1.0 \
  --tiled \
  --out /raid/workspace1/lc/HumanAndRobot/Output_Inference/wan21_1.3b/cloth_dit+vace.mp4

# # T2V（不提供 --vace_video 时自动退化）
# CUDA_VISIBLE_DEVICES=7 \
# python infer_dit_plus_vace_lora.py \
#   --base_dir /raid/share_model/Wan2.1-VACE-1.3B \
#   --dit_lora  /path/to/dit_lora.safetensors \
#   --vace_lora /path/to/vace_lora.safetensors \
#   --prompt "a robotic arm folds a T-shirt on a wooden table, soft key light" \
#   --width 832 --height 480 --num_frames 81 --fps 24 \
#   --out ./t2v_dit+vace.mp4
