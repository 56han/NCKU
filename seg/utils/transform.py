import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),                                                                        # 水平翻轉，概率為 0.5
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=1),                  # 平移、縮放、旋轉
        A.LongestMaxSize(max_size=256, p=1),                                                            # 調整長邊大小為 256
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),    # 補齊到 256x256
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),                                                    # 加性高斯噪聲，概率為 0.2
        A.Normalize(mean=0.5, std=0.5),                                                                 # 正規化
        ToTensorV2(),                                                                                   # 轉換為 Tensor
    ])

def get_val_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=256, p=1),                                                            # 調整長邊大小為 256
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),    # 補齊到 256x256
        A.Normalize(mean=0.5, std=0.5),                                                                 # 正規化
        ToTensorV2(),                                                                                   # 轉換為 Tensor
    ])