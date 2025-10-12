CUDA_VISIBLE_DEVICES=4 \
python infer_dit_lora.py \
  --base_dir /raid/share_model/Wan2.1-VACE-1.3B \
  --lora_path /raid/workspace1/lc/HumanAndRobot/Output_Model/Wan2.1-VACE-1.3B_dit_lora/push_plate/epoch-1.safetensors \
  --prompt "The mechanical arm push a plate" \
  --width 832 --height 480 --num_frames 81 --fps 24 \
  --alpha 1.0 --out /raid/workspace1/lc/HumanAndRobot/Output_Inference/wan21_1.3b/push_plate_demo.mp4 \
  --tiled
