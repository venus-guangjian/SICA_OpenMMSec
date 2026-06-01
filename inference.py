import argparse
import torch
from PIL import Image
from model import CLIP_LORA_PURE


def main():
    parser = argparse.ArgumentParser(description="Inference script for CLIP_LORA_PURE")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weight", type=str, required=True, help="Path to model weights (.pt or .pth)")
    parser.add_argument("--model_name", type=str, default="ViT-L/14", help="Name of the backbone (e.g. ViT-L/14, RN50)")
    args = parser.parse_args()
    # /mnt/data0/public_datasets/OpenMMSecV2/AIGC/latent_diffusion/stable_diffusion/imgs/be0d89c54f924cd68f999501ace68bfa.png
    # /mnt/data0/dubo/workspace/ForensicHub/log/OpenMMSecV2_V4/cliplora_pure_lora_train/checkpoint-9.pth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型
    print(f"Loading model architecture: {args.model_name}")
    model = CLIP_LORA_PURE(name=args.model_name, num_classes=1, tune_mode="lora")

    # 2. 加载权重
    print(f"Loading weights from {args.weight}")
    checkpoint = torch.load(args.weight, map_location="cpu", weights_only=False)

    # 兼容常见的 checkpoint 保存格式 (直接保存 state_dict，或者嵌套在 'model'/'state_dict' 键中)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 3. 处理图片 (直接调用模型类内部保存的 preprocess)
    print(f"Reading and preprocessing image: {args.image}")
    try:
        img = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 添加 batch 维度 [1, C, H, W] 并传到设备
    img_tensor = model.preprocess(img).unsqueeze(0).to(device)

    # 4. 推理
    with torch.no_grad():
        output = model(image=img_tensor)

    # 取出预测概率 (shape 通常是 [1] 或标量)
    prob = output["pred_label"].item()

    print("-" * 30)
    print(f"Prediction Probability: {prob:.6f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
