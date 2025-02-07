import subprocess
import itertools
from pathlib import Path

def run_batch_experiments():
    batch_sizes = [8, 16]
    learning_rates = [1e-4, 3e-4]
    epochs_list = [100]
    encoder_names = ["resnet50", "efficientnet-b0", "efficientnet-b1" , "efficientnet-b2"]
    schedulers = ["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"]
    model_names = ["Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN"]

    # 結果保存的主文件夾
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    experiments = list(itertools.product(batch_sizes, learning_rates, epochs_list, encoder_names, schedulers, model_names))

    for i, (batch_size, learning_rate, epochs, encoder_name, scheduler, model_name) in enumerate(experiments):
        print(f"Running experiment {i + 1}/{len(experiments)}...")

        # 構建命令
        cmd = [
            "python", "train.py",
            "--batch_size", str(batch_size),
            "--learning_rate", str(learning_rate),
            "--epochs", str(epochs),
            "--encoder_name", encoder_name,
            "--scheduler", scheduler,
            "--model", model_name,
            "--crop_img" # 訓練時使用裁剪的圖像
        ]

        # 日誌文件名
        log_file = log_dir / f"experiment_{i + 1}_batch{batch_size}_lr{learning_rate}_epochs{epochs}_encoder{encoder_name}_scheduler{scheduler}.txt"

        # 執行命令並將輸出寫入日誌文件
        with log_file.open("w") as f:
            f.write(f"Experiment {i + 1}/{len(experiments)}\n")
            f.write(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}, Encoder: {encoder_name}, Scheduler: {scheduler}\n\n")
            subprocess.run(cmd, stdout=f, stderr=f, text=True)

        print(f"Experiment {i + 1} completed. Log saved to {log_file}")


if __name__ == "__main__":
    run_batch_experiments()
