import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

from pathlib import Path
from tqdm import tqdm 
import datetime

from utils.data_loading import UltrasoundDataset
from utils.metrics import dice_score
from utils.save_result import save_training_results
from utils.transform import get_train_transforms, get_val_transforms
import argparse

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    dice_scores = []
    loop = tqdm(loader, desc="Training", leave=False)
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        preds = (outputs > 0.5).float()
        dice = dice_score(preds, masks)
        dice_scores.append(dice)
        
        loop.set_postfix(loss=loss.item(), dice_score=dice)
    avg_loss = epoch_loss / len(loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    return avg_loss, avg_dice

def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0
    dice_scores = []
    loop = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            # 計算 Dice Score
            preds = (outputs > 0.5).float()
            dice = dice_score(preds, masks)
            dice_scores.append(dice)
            # 更新進度條信息
            loop.set_postfix(loss=loss.item(), dice_score=dice)
    avg_loss = val_loss / len(loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    return avg_loss, avg_dice

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

def choose_scheduler(optimizer, scheduler_type, epochs):
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return scheduler

def main(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size, learning_rate, epochs, encoder_name, scheduler_type, model_name, crop_img_flag):
    train_img_dir = Path(train_img_dir)
    train_mask_dir = Path(train_mask_dir)
    val_img_dir = Path(val_img_dir)
    val_mask_dir = Path(val_mask_dir)

    # Early Stopping 
    patience = 10  # 容忍多少個 epoch 沒有改善
    patience_counter = 0  # 記錄目前沒有改善的 epoch 次數
    best_val_dice = 0.0  # 初始化最佳驗證 Dice Score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    train_dataset = UltrasoundDataset(train_img_dir, train_mask_dir, transform=train_transforms)
    val_dataset = UltrasoundDataset(val_img_dir, val_mask_dir, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = choose_model(model_name, encoder_name)
    model = model.to(device)

    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler
    scheduler = choose_scheduler(optimizer, scheduler_type, epochs)
    scheduler_name = type(scheduler).__name__

    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    if crop_img_flag:
        ckpt_dir = Path(f"./ckpt/crop/{model_name}/{timestamp}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_dir = Path(f"./ckpt/no_crop/{model_name}/{timestamp}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate_one_epoch(model, val_loader, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_scores.append(train_dice)
        val_dice_scores.append(val_dice)

        scheduler.step(val_loss if scheduler_name == "ReduceLROnPlateau" else None)

        # 檢查是否為最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0  # 重置 Early Stopping 計數器
            # 保存最佳模型
            best_model_path = Path(f"{ckpt_dir}/best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Val Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1

        # Early Stopping 檢查
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Save training results
    save_training_results(ckpt_dir, train_losses, val_losses, train_dice_scores, val_dice_scores,
                          timestamp, batch_size, learning_rate, epochs, encoder_name, scheduler_name)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run U-Net training with different configurations.")

    # parser.add_argument("--train_img_dir", type=str, default="./data/crop/train_imgs/", help="Path to training images")
    # parser.add_argument("--train_mask_dir", type=str, default="./data/crop/train_masks/", help="Path to training masks")
    # parser.add_argument("--val_img_dir", type=str, default="./data/crop/val_imgs/", help="Path to validation images")
    # parser.add_argument("--val_mask_dir", type=str, default="./data/crop/val_masks/", help="Path to validation masks")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b0", 
                        help="Encoder name (e.g., efficientnet-b0, resnet34, etc.)")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", 
                        choices=["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"], 
                        help="Scheduler type to adjust learning rate")
    parser.add_argument("--model", type=str, default="Unet", 
                        choices=["Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN"], 
                        help="Model architecture to use (e.g., Unet, UnetPlusPlus, DeepLabV3Plus, FPN, PSPNet)")
    parser.add_argument("--crop_img", action="store_true", help="Train on cropped images") # 執行時沒有"--crop_img"代表沒有使用 crop img

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.crop_img:
        train_img_dir = "./data/crop/train_imgs/"
        train_mask_dir = "./data/crop/train_masks/"
        val_img_dir = "./data/crop/val_imgs/"
        val_mask_dir = "./data/crop/val_masks/"
    else:
        train_img_dir = "./data/no_crop/train_imgs/"
        train_mask_dir = "./data/no_crop/train_masks/"
        val_img_dir = "./data/no_crop/val_imgs/"
        val_mask_dir = "./data/no_crop/val_masks/"

    main(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        encoder_name=args.encoder_name,
        scheduler_type=args.scheduler,
        model_name=args.model,
        crop_img_flag=args.crop_img
    )