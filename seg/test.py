import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import segmentation_models_pytorch as smp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2

from utils.data_loading import UltrasoundDataset  
from utils.metrics import dice_score, iou_score  
from utils.transform import get_val_transforms
from utils.save_result import save_excel_results, save_overlay_image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test trained segmentation model on test dataset.")
    
    # 測試數據的路徑
    parser.add_argument("--test_img_dir", type=str, default="./data/crop/test_imgs/", help="Path to test images")
    parser.add_argument("--test_mask_dir", type=str, default="./data/crop/test_masks/", help="Path to test masks")
    
    # 模型加載和配置
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (best_model.pth)")
    parser.add_argument("--model", type=str, default="Unet", 
                        choices=["Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN"], 
                        help="Model architecture used for training")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b0", 
                        help="Encoder name (e.g., efficientnet-b0, resnet50, etc.)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--save_img", action="store_true", help="Save overlay images")
    parser.add_argument("--save_excel", action="store_true", help="Save test results to Excel")
    # store_true 一旦有這個參數，做出動作 “ 將其值標為True ”，也就是沒有時，默認狀態下其值為 False。
    
    return parser.parse_args()

def choose_model(model_name, encoder_name):
    if model_name == "Unet":
        model = smp.Unet(encoder_name, encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(encoder_name, encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(encoder_name, encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "FPN":
        model = smp.FPN(encoder_name, encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "PSPNet":
        model = smp.PSPNet(encoder_name, encoder_weights="imagenet", in_channels=1, classes=1)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model

def main(test_img_dir, test_mask_dir, model_path, model_name, encoder_name, batch_size, save_img_flag, save_excel_flag):
    test_img_dir = Path(test_img_dir)
    test_mask_dir = Path(test_mask_dir)

    output_dir = Path(f"./test_results/{model_name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert test_img_dir.exists(), f"Test image directory {test_img_dir} does not exist."
    assert test_mask_dir.exists(), f"Test mask directory {test_mask_dir} does not exist."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 測試數據集
    transform = get_val_transforms
    test_dataset = UltrasoundDataset(test_img_dir, test_mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加載模型
    model = choose_model(model_name, encoder_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 記錄每張影像的指標
    results = []
    loop = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()  # 二值化處理
            
            for i in range(images.size(0)):
                # 提取單張影像、預測和真實標註
                img = images[i].cpu().numpy().squeeze()
                true_mask = masks[i].cpu().numpy().squeeze()
                pred_mask = preds[i].cpu().numpy().squeeze()

                dice = dice_score(torch.tensor(pred_mask), torch.tensor(true_mask))
                iou = iou_score(torch.tensor(pred_mask), torch.tensor(true_mask))

                img_name = Path(test_dataset.image_paths[batch_idx * batch_size + i]).name
                results.append({
                    "Image Name": img_name,
                    "Dice Score": dice,
                    "IoU Score": iou
                })

                # 繪製重疊圖像
                if save_img_flag:
                    overlay_path = output_dir / f"{img_name}.png"
                    save_overlay_image(img, true_mask, pred_mask, overlay_path)

    # 將結果存入 Excel
    if save_excel_flag:     
        excel_path = "test_results/all_test.xlsx"
        save_excel_results(results, model_name, excel_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(
        test_img_dir=args.test_img_dir,
        test_mask_dir=args.test_mask_dir,
        model_path=args.model_path,
        model_name=args.model,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        save_img_flag = args.save_img,
        save_excel_flag = args.save_excel
    )
