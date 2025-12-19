import os
import subprocess

# ----------------------------
# 基本路径配置
# ----------------------------
CUDA_DEVICE = "6"

INFER_SCRIPT = "/home/liusheng/worksapce2024/github/Real-ESRGAN/inference_realesrgan_cbct.py"
INPUT_DIR = "/home/liusheng/worksapce2024/github/Real-ESRGAN/datasets/CBCT/test_set/cbct_input"
OUTPUT_BASE = "/home/liusheng/worksapce2024/github/Real-ESRGAN/datasets/CBCT/test_set"

MODEL_NAME = "RealESRGAN_x1plus_cbct"
OUTSCALE = "1"



# ----------------------------
# 主逻辑
# ----------------------------
def run_inference(INFER_SCRIPT, CHECKPOINTS):
    for ckpt_path in CHECKPOINTS:
        # ckpt_path = os.path.join(MODEL_DIR, ckpt)
        head, ckpt = os.path.split(ckpt_path)
        head2, _ = os.path.split(head)
        head3, model_name = os.path.split(head2)
        # train_RealESRNetx2plus_7k_cbct_rescale0to1_epoch10000
        if 'train_RealESRNetx2plus_7k_cbct_rescale0to1' in model_name:
            model_name = 'rescale0to1_net_out'
        model_name = model_name.replace("train_RealESRGANx1plus_50K_gray_from_zero_cbct_", "")
        
        ckpt_tag = ckpt.replace(".pth", "")
        ckpt_tag2 = ckpt_tag.split('_')[-1]

        output_dir = os.path.join(OUTPUT_BASE, f"{model_name}_epoch{ckpt_tag2}")

        # 检查模型文件
        if not os.path.isfile(ckpt_path):
            print(f"[SKIP] Model not found: {ckpt_path}")
            continue

        # 创建输出目录
        if os.path.exists(output_dir):
            continue

        # 组装命令
        cmd = [
            "bash",
            "-c",
            (
                f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} "
                f"python {INFER_SCRIPT} "
                f"--model_name {MODEL_NAME} "
                f"--outscale {OUTSCALE} "
                f"--model_path {ckpt_path} "
                f"--input {INPUT_DIR} "
                f"--output {output_dir}"
            )
        ]
        
        

        print(f"\n[RUN] Inference with:  {model_name} -> {ckpt}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    INFER_SCRIPT = "/home/liusheng/worksapce2024/github/Real-ESRGAN/inference_realesrgan_cbct.py"
    # 需要推理的 checkpoint
    CHECKPOINTS = [
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_default_loss/models/net_g_10000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_default_loss/models/net_g_20000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_default_loss/models/net_g_30000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_default_loss/models/net_g_40000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_default_loss/models/net_g_50000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_l1_gan_loss/models/net_g_30000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_l1_gan_loss/models/net_g_20000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_l1_gan_loss/models/net_g_40000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_l1_gan_loss/models/net_g_50000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_l1_gan_loss/models/net_g_10000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_only_l1_loss/models/net_g_30000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_only_l1_loss/models/net_g_20000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_only_l1_loss/models/net_g_40000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_only_l1_loss/models/net_g_50000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRGANx1plus_50K_gray_from_zero_cbct_only_l1_loss/models/net_g_10000.pth",
    ]   
    run_inference(INFER_SCRIPT, CHECKPOINTS)
    
    
    INFER_SCRIPT2 = "/home/liusheng/worksapce2024/github/Real-ESRGAN/inference_realesrgan_cbct_0to1.py"
    # 需要推理的 checkpoint
    CHECKPOINTS2 = [
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRNetx2plus_7k_cbct_rescale0to1/models/net_g_50000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRNetx2plus_7k_cbct_rescale0to1/models/net_g_10000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRNetx2plus_7k_cbct_rescale0to1/models/net_g_40000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRNetx2plus_7k_cbct_rescale0to1/models/net_g_20000.pth",
        "/home/liusheng/worksapce2024/github/Real-ESRGAN/experiments/train_RealESRNetx2plus_7k_cbct_rescale0to1/models/net_g_30000.pth",
    ]   
    run_inference(INFER_SCRIPT2, CHECKPOINTS2)
