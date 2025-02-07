import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import pandas as pd

def plot_training_metrics(epochs, train_losses, val_losses, train_dice_scores, val_dice_scores, save_path):
    actual_epochs = len(train_losses)  # 根據實際數據長度計算
    epochs_range = range(1, actual_epochs + 1)

    plt.figure(figsize=(12, 5))

    # 損失圖
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Dice Score 圖
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_dice_scores, label='Train Dice Score')
    plt.plot(epochs_range, val_dice_scores, label='Val Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Dice Score over Epochs')
    plt.legend()

    plt.savefig(save_path)
    # plt.show()

def save_training_results(ckpt_dir, train_losses, val_losses, train_dice_scores, val_dice_scores, timestamp, batch_size, learning_rate, epochs, encoder_name, scheduler_name):
    plot_path = ckpt_dir / f"{timestamp}.png"
    plot_training_metrics(epochs, train_losses, val_losses, train_dice_scores, val_dice_scores, plot_path)

    best_val_dice = max(val_dice_scores)
    summary_path = ckpt_dir / f"{timestamp}.txt"
    with summary_path.open("w") as f:
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Encoder: {encoder_name}\n")
        f.write(f"Scheduler: {scheduler_name}\n")
        f.write(f"Best Validation Dice Score: {best_val_dice:.4f}\n")
        
    print(f"Training results saved to {ckpt_dir}")

def save_excel_results(results, model_name, excel_path):
    df = pd.DataFrame(results)
    sheet = model_name
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Results saved to {excel_path}")

def save_overlay_image(img, true_mask, pred_mask, save_path):
    # 確保 true_mask 和 pred_mask 為整數類型
    true_mask = (true_mask > 0.5).astype(np.uint8)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    overlay = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
    
    overlap = true_mask & pred_mask
    overlay[..., 0] = pred_mask * 255  # 紅色
    overlay[..., 1] = true_mask * 255  # 綠色
    overlay[overlap == 1] = [255, 255, 0]  # 重疊區域設置為黃色

    # 原始影像縮放到 [0, 255]
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    img_colored = np.stack([img, img, img], axis=-1)

    combined = cv2.addWeighted(img_colored, 0.7, overlay, 0.3, 0)

    plt.imsave(save_path, combined)