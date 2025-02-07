
import pandas as pd
import os
from pathlib import Path


def read_txt_with_debug(model_name):
    all_data = []
    base_dir = Path(f"../ckpt/crop/{model_name}")

    try:
        # Check if the base directory exists
        if not os.path.exists(base_dir):
            print(f"Error: Base directory '{base_dir}' does not exist.")
            return

        # Loop through each subdirectory in the base directory
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(folder_path):  # Ensure it's a directory
                txt_file = os.path.join(folder_path, f"{folder_name}.txt")
                if os.path.isfile(txt_file):  # Ensure the txt file exists
                    data_dict = {"folder_name": folder_name}
                    try:
                        with open(txt_file, 'r') as file:
                            for line in file:
                                if ':' in line:
                                    key, value = line.strip().split(':', 1)
                                    data_dict[key.strip()] = value.strip()
                        all_data.append(data_dict)
                    except Exception as e:
                        print(f"Error reading file '{txt_file}': {e}")
                else:
                    print(f"Warning: File '{txt_file}' does not exist.")
            else:
                print(f"Warning: '{folder_path}' is not a directory.")

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(all_data)

        # Define the output file path for Excel
        output_file = Path(f"../train_result/{model_name}_para_dice.xlsx")

        try:
            # Save the DataFrame to Excel
            df.to_excel(output_file, index=False)
            print(f"Excel file successfully saved to '{output_file}'")
        except Exception as e:
            print(f"Error writing to Excel file '{output_file}': {e}")

    except Exception as e:
        print(f"Unexpected error during execution: {e}")

if __name__ == "__main__":
    model_names = ["Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN"]

    # Run the function
    for model_name in model_names:
        read_txt_with_debug(model_name)

    # 當使用 crop_img 時，要修改 base_dir 


