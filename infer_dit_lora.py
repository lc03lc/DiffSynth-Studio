#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

def main():
    parser = argparse.ArgumentParser(description="Wan2.1 (VACE-1.3B) + DiT LoRA 推理 - T2V")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="预训练模型目录（Wan2.1-VACE-1.3B）")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="DiT LoRA 权重 .safetensors 路径（例如 epoch-xxx.safetensors）")
    parser.add_argument("--prompt", type=str, required=True,
                        help="文本提示词")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0, help="DiT LoRA 强度（0~1）")
    parser.add_argument("--out", type=str, default="wan_dit_lora_t2v.mp4")
    parser.add_argument("--tiled", action="store_true", help="启用分块采样以省显存（高分辨率推荐）")
    args = parser.parse_args()

    # 选择 dtype（bf16 可用则优先）
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = args.base_dir.rstrip("/")

    # ---- 构建管线（Wan2.1-VACE-1.3B）----
    # 1.3B 的 diffusion 是单文件；仍需用 ModelConfig 包一下
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(path=[f"{base}/diffusion_pytorch_model.safetensors"]),
            ModelConfig(path=f"{base}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{base}/Wan2.1_VAE.pth"),
        ],
    )

    # ---- 挂载 DiT LoRA（只挂到主干）----
    pipe.load_lora(pipe.dit, args.lora_path, alpha=args.alpha)

    # 建议开启显存管理（会自动在前向不同阶段做参数/激活迁移）
    if hasattr(pipe, "enable_vram_management"):
        pipe.enable_vram_management()

    # ---- 生成 ----
    kwargs = dict(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        seed=args.seed,
    )
    # 分块采样可在高分辨率下减少显存
    if args.tiled:
        kwargs["tiled"] = True

    print(f"[INFO] dtype={dtype}, device={device}")
    print(f"[INFO] prompt={args.prompt}")
    print(f"[INFO] size={args.width}x{args.height}, frames={args.num_frames}, fps={args.fps}")
    print(f"[INFO] DiT LoRA={args.lora_path}, alpha={args.alpha}")

    video = pipe(**kwargs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_video(video, args.out, fps=args.fps, quality=5)
    print(f"[DONE] 保存到: {args.out}")

if __name__ == "__main__":
    main()
