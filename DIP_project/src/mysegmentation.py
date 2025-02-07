import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from UNet.unet.unet_model import UNet
import cv2
import torch.nn as nn
import torch
from UNet.utils.dice_score import multiclass_dice_coeff, dice_coeff
import matplotlib.pyplot as plt
from UNet.utils.data_loading import BasicDataset
import math
import json
from sklearn.metrics import f1_score

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)


    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def get_output_filenames(output):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return output or list(map(_generate_name, input))

import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_img_and_mask(name, img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img, cmap='gray')
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i, cmap='gray')


def label_to_xyxy(label, image_size):
    class_index, x_center, y_center, width, height = label
    x_min = int((float(x_center) - float(width) / 2)*image_size)
    y_min = int((float(y_center) - float(height) / 2)*image_size)
    x_max = int((float(x_center) + float(width) / 2)*image_size)
    y_max = int((float(y_center) + float(height) / 2)*image_size)

    return [class_index, x_min, y_min, x_max, y_max]

# (B, G, R)
class_color = {
    # 'warthin': (255, 0, 0, 128), 
    # 'mix': (0, 255, 255, 128) ,
    # 'cancer': (0, 255, 0, 128) 
    'Warthin': (0, 0, 255), # red
    'Mix': (255, 255, 0) , # blue
    'Cancer': (0, 255, 0) # green
}

def overlap_img(path, txt_path, origin_img_path, mask, i, left_class, right_class):
    save_path = './output_crop_img/predict_output'
    crop_path = './output_crop_img/images'
    mask_path = './segmentation/predict'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(crop_path):
        os.makedirs(crop_path, exist_ok=True)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path, exist_ok=True)
    
    image_name = os.path.basename(origin_img_path)
    image_filename = image_name.replace('.png', '')
    json_to_GT(path, image_name)

    black_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_np = np.array(mask)
    mask_modified = mask_np.copy()
    mask_np_8_bit = cv2.convertScaleAbs(mask_modified)
    mask_np_3_channel = cv2.cvtColor(mask_np_8_bit, cv2.COLOR_GRAY2RGB)
    mask_np_3_channel *= 255
    white_pixels = np.all(mask_np_3_channel == [255, 255, 255], axis=-1)  # 找出值為白色的像素位置

    if left_class == 'Normal' and right_class != 'Normal':
        left_class = right_class
    if right_class == 'Normal' and left_class != 'Normal':
        right_class = left_class


    # mask = np.stack(mask, mask, mask)
    with open(txt_path, 'r') as file:
        content = file.readlines()
    original_image = cv2.imread(origin_img_path)
    image_path = os.path.join(crop_path, image_filename +'_'+ str(i) +'.png')
    img_size = 512
    combined_image = original_image.copy()  # 創建一個原始圖像的副本
    bbox_position = {}

    
    for line in content:
        data = line.split()  # 按空格分割每行數據
        class_index,min_x, min_y, max_x, max_y = label_to_xyxy(data, img_size)
        bbox_position[class_index]= [int(min_x), int(min_y), int(max_x), int(max_y)]
    
    alpha = 0.75
    L_file_path = os.path.join(save_path, image_filename + '_0.png')
    black_file_path = os.path.join(mask_path, image_filename + '_mask.png')

    if i == 0:
        class_index = '0'
        min_x, min_y, max_x, max_y = bbox_position[class_index]
        x = max_x - min_x
        y = max_y - min_y
        
        # 貼回黑底
        black_image[math.floor(min_y):math.ceil(min_y + y), math.floor(min_x):math.ceil(min_x + x)] = mask_np_3_channel
        cv2.imwrite(black_file_path,black_image)

        # 貼回小圖
        mask_np_3_channel[white_pixels] = class_color[left_class]
        L_image = cv2.imread(image_path)
        mask_np_3_channel = mask_np_3_channel.astype(np.uint8)
        L_result = cv2.addWeighted(L_image, 1, mask_np_3_channel, 1 - alpha, 0)

        # 貼回原圖
        combined_image[math.floor(min_y):math.ceil(min_y + y), math.floor(min_x):math.ceil(min_x + x)] = L_result
        cv2.imwrite(L_file_path,combined_image)

    elif i == 1:
        class_index = '1'
        min_x, min_y, max_x, max_y = bbox_position[class_index]
        x = max_x - min_x
        y = max_y - min_y
        # 貼回黑底
        black_image = cv2.imread(black_file_path)
        black_image[math.floor(min_y):math.ceil(min_y + y), math.floor(min_x):math.ceil(min_x + x)] = mask_np_3_channel
        cv2.imwrite(black_file_path,black_image)
        
        # 貼回小圖
        mask_np_3_channel[white_pixels] = class_color[right_class]
        R_image = cv2.imread(image_path)
        mask_np_3_channel = mask_np_3_channel.astype(np.uint8)
        R_result = cv2.addWeighted(R_image, 1, mask_np_3_channel, 1 - alpha, 0)

        # 貼回原圖
        combined_image = cv2.imread(L_file_path)
        combined_image[math.floor(min_y):math.ceil(min_y + y), math.floor(min_x):math.ceil(min_x + x)] = R_result
        R_file_path = os.path.join(save_path, image_filename + '_1.png')
        cv2.imwrite(R_file_path, combined_image)

def json_to_GT(path, image_filename):
    # input_folder = './demo_test9'
    input_folder = path
    output_folder = './segmentation/GT'  # 輸出目錄

    # 創建輸出資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    image_path = input_folder 
    json_path = input_folder.replace('.png', '.json')
    # json_name = image_filename.replace('.png', '.json')
    # json_path = os.path.join(input_folder, json_name)
    # image_path = os.path.join(input_folder, image_filename)
    
    # 讀取 JSON 檔案
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 創建一個和圖像大小相同的全黑圖像
    image = Image.new('L', (data['imageWidth'], data['imageHeight']), 0)
    draw = ImageDraw.Draw(image)
    GT_label = ''

    # 繪製每個形狀（即多邊形）
    for shape in data['shapes']:
        label = shape['label']
        if label == "left normal" or label == "right normal":
            continue
        GT_label =label
        points = shape['points']
        polygon_points = [tuple(point) for point in points]
        draw.polygon(polygon_points, fill=255)  # 在 mask 圖像上繪製多邊形
        
        # 將 PIL 圖像轉換為 NumPy array
        mask_np = np.array(image)

        # 輸出的 mask 檔案路徑
        output_mask_path = os.path.join(output_folder, image_filename.split('.')[0]+ '_GT.png')

        # 將 NumPy array 保存為 PNG 檔案
        Image.fromarray(mask_np).save(output_mask_path)


    GT_label = GT_label.capitalize()
    alpha = 0.75

    mask_np_3_channel = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
    white_pixels = np.all(mask_np_3_channel == [255, 255, 255], axis=-1)  # 找出值為白色的像素位置
    mask_np_3_channel[white_pixels] = class_color[GT_label]
    
    original_img = cv2.imread(image_path)

    mask_np_3_channel = mask_np_3_channel.astype(np.uint8)
    result = cv2.addWeighted(original_img, 1, mask_np_3_channel, 1 - alpha, 0)
    output_mask_path = os.path.join(output_folder, image_filename.split('.')[0]+ '_com.png')
    cv2.imwrite(output_mask_path, result)

def mypredict(model, input, output, viz, no_save, mask_threshold, scale, bilinear, classes, original_img_path, left_class, right_class,path):
    net = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    
    dice_score_list = []
    in_files = input
    out_files = get_output_filenames(output)
    for i, filename in enumerate(in_files):
        dice_score = 0
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device)
        

        criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        # 貼回原圖 
        image_name = os.path.basename(original_img_path)
        predict_txt_path = os.path.join("./ObjectDetection/detect/exp/labels", f"{image_name.split('.')[0]}.txt")
        overlap_img(path, predict_txt_path, original_img_path, mask, i, left_class, right_class)
        
    # 再計算dice
    GT_masks_folder = r'segmentation\GT'
    GT_mask_file_path = os.path.join(".",GT_masks_folder, f"{image_name.split('.')[0]}_GT.png")
    GT_mask = cv2.imread(GT_mask_file_path)
    GT_mask_flat = GT_mask.flatten()/255
    # print(np.unique(GT_mask_flat))

    predict_masks_folder = r'segmentation\predict'
    predict_mask_file_path = os.path.join(".",predict_masks_folder, f"{image_name.split('.')[0]}_mask.png")
    predict_mask = cv2.imread(predict_mask_file_path)
    predicted_mask_flat = predict_mask.flatten()/255
    # print(np.unique(predicted_mask_flat))

    # dice_coefficient_f1 = f1_score(GT_mask_flat, predicted_mask_flat, labels=[0, 255], average='binary')
    # print(dice_coefficient_f1)

    dice_coefficient = (2 * np.sum(predicted_mask_flat * GT_mask_flat)) / (np.sum(predicted_mask_flat) + np.sum(GT_mask_flat))
    # print(2 * np.sum(predicted_mask_flat * GT_mask_flat))
    # print(np.sum(predicted_mask_flat) + np.sum(GT_mask_flat))
    # print(dice_coefficient)
    return dice_coefficient