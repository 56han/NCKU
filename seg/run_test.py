import pandas as pd
import subprocess
from pathlib import Path

def run_tests_from_excel(excel_path, test_img_dir, test_mask_dir):
    # 讀取 Excel 文件
    data = pd.read_excel(excel_path, sheet_name='工作表1')

    # 測試日誌保存目錄
    log_dir = Path("./test_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 遍歷每一行配置進行測試
    for idx, row in data.iterrows():
        model_name = row['model_name']
        encoder_name = row['Encoder']
        batch_size = row['Batch Size']
        folder_name = row['folder_name']
        model_path = Path(f"./ckpt/{model_name}/{folder_name}/best_model.pth")

        # 確保模型文件存在
        if not model_path.exists():
            print(f"Model file {model_path} does not exist. Skipping this test.")
            continue

        # 日誌文件名稱
        log_file = log_dir / f"test_{model_name}_{folder_name}.txt"

        # 測試命令
        cmd = [
            "python", "test.py",
            "--test_img_dir", test_img_dir,
            "--test_mask_dir", test_mask_dir,
            "--model_path", str(model_path),
            "--model", model_name,
            "--encoder_name", encoder_name,
            "--batch_size", str(batch_size),
            "--save_img"
            # "--save_excel"
        ]

        # 執行命令並將輸出記錄到日誌文件
        with log_file.open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=log, text=True)

if __name__ == "__main__":
    test_img_dir = "./data/crop/test_imgs/"
    test_mask_dir = "./data/crop/test_masks/"
    excel_path = "./result/best.xlsx"

    run_tests_from_excel(excel_path, test_img_dir, test_mask_dir)
