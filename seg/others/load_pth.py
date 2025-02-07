import torch

# 載入 .pth 文件
checkpoint = torch.load('model_path.pth')

# 檢查內容
for key, value in checkpoint.items():
    print(f"Key: {key}, Shape: {value.shape}")


"""
如果你存的是 model.state_dict()，會看到類似以下的輸出：

Key: layer1.weight, Shape: torch.Size([64, 3, 3, 3])
Key: layer1.bias, Shape: torch.Size([64])
Key: layer2.weight, Shape: torch.Size([128, 64, 3, 3])
Key: layer2.bias, Shape: torch.Size([128])

"""

"""
你可以存儲一個更大的字典，例如：

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': current_epoch,
    'loss': loss_value
}, 'model_path.pth')

checkpoint = torch.load('model_path.pth')
print(checkpoint.keys())  # 查看有哪些鍵

"""