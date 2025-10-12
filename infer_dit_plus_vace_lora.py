#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import cv2
import numpy as np
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def parse_args():
    ap = argparse.ArgumentParser(
        description="Wan2.1 VACE 推理（同时挂载 DiT LoRA + VACE LoRA，并输出 GT+生成拼接视频）"
    )
    # 基座模型
    ap.add_argument("--base_dir", type=str, required=True,
                    help="预训练模型目录（Wan2.1-VACE-1.3B 或 14B）")
    ap.add_argument("--is_14b", action="store_true",
                    help="若使用 Wan2.1-VACE-14B 则加此开关（会按 7 片加载）")

    # LoRA
    ap.add_argument("--dit_lora", type=str, required=True,
                    help="DiT LoRA .safetensors 路径")
    ap.add_argument("--vace_lora", type=str, required=True,
                    help="VACE LoRA .safetensors 路径")
    ap.add_argument("--alpha_dit", type=float, default=0.6,
                    help="DiT LoRA 强度（0~1）")
    ap.add_argument("--alpha_vace", type=float, default=1.0,
                    help="VACE LoRA 强度（0~1）")

    # 任务 & 输入
    ap.add_argument("--prompt", type=str, required=True,
                    help="文本提示词")
    ap.add_argument("--vace_video", type=str, default="",
                    help="作为条件输入的视频路径（不填则做 T2V）")
    ap.add_argument("--vace_reference_image", type=str, default="",
                    help="参考图像，支持逗号分隔多张")
    ap.add_argument("--vace_video_mask", type=str, default="",
                    help="蒙版视频/图（可选）")

    # 生成配置
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiled", action="store_true",
                    help="分块采样，省显存（高分辨率建议开启）")

    # 输出
    ap.add_argument("--out", type=str, default="out_dit+vace.mp4")

    return ap.parse_args()


def build_pipeline(base_dir: str, is_14b: bool, dtype, device):
    base = base_dir.rstrip("/")

    if is_14b:
        # Wan2.1-VACE-14B（7 片 diffusion）
        diff_list = [
            f"{base}/diffusion_pytorch_model-00001-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00002-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00003-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00004-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00005-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00006-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00007-of-00007.safetensors",
        ]
    else:
        # Wan2.1-VACE-1.3B（单文件 diffusion）
        diff_list = [f"{base}/diffusion_pytorch_model.safetensors"]

    return WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(path=diff_list),
            ModelConfig(path=f"{base}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{base}/Wan2.1_VAE.pth"),
        ],
    )


def load_video(path, max_frames=None, width=None, height=None):
    """把 mp4 解码成 numpy 图像列表 (H,W,C, uint8)，可选 resize"""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if width and height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def main():
    args = parse_args()

    # 选择 dtype
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 基座
    pipe = build_pipeline(args.base_dir, args.is_14b, dtype, device)

    # 同时加载两份 LoRA
    pipe.load_lora(pipe.dit,  args.dit_lora,  alpha=args.alpha_dit)
    pipe.load_lora(pipe.vace, args.vace_lora, alpha=args.alpha_vace)

    if hasattr(pipe, "enable_vram_management"):
        pipe.enable_vram_management()

    # 组织推理参数
    infer_kwargs = dict(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        seed=args.seed,
    )
    if args.tiled:
        infer_kwargs["tiled"] = True

    # VACE 条件（有就加）
    if args.vace_video:
        frames = load_video(
            args.vace_video,
            max_frames=args.num_frames,
            width=args.width,
            height=args.height,
        )
        infer_kwargs["vace_video"] = frames

    if args.vace_reference_image:
        refs = [p.strip() for p in args.vace_reference_image.split(",") if p.strip()]
        if len(refs) == 1:
            infer_kwargs["vace_reference_image"] = refs[0]
        elif len(refs) > 1:
            infer_kwargs["vace_reference_image"] = refs

    if args.vace_video_mask:
        infer_kwargs["vace_video_mask"] = args.vace_video_mask

    # 打印关键信息
    print(f"[INFO] device={device}, dtype={dtype}")
    print(f"[INFO] base={args.base_dir}, 14B={args.is_14b}")
    print(f"[INFO] DIT LoRA={args.dit_lora} (alpha={args.alpha_dit})")
    print(f"[INFO] VACE LoRA={args.vace_lora} (alpha={args.alpha_vace})")
    if args.vace_video:
        print(f"[INFO] V2V 模式: vace_video={args.vace_video}")
    else:
        print(f"[INFO] T2V 模式（未提供 vace_video）")

    # ========= 生成 =========
    video_gen = pipe(**infer_kwargs)

    # ========= 拼接原始视频 + 生成视频 =========
    if args.vace_video:
        video_gt = load_video(
            args.vace_video,
            max_frames=args.num_frames,
            width=args.width,
            height=args.height,
        )

        # 对齐帧数
        min_len = min(len(video_gt), len(video_gen))
        video_gt = video_gt[:min_len]
        video_gen = video_gen[:min_len]

        # 水平拼接： [GT | 生成]
        video_concat = [np.concatenate([gt, gen], axis=1)
                        for gt, gen in zip(video_gt, video_gen)]
    else:
        video_concat = video_gen

    # ========= 保存 =========
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_video(video_concat, args.out, fps=args.fps, quality=5)
    print(f"[DONE] 保存到: {args.out}")


if __name__ == "__main__":
    main()
