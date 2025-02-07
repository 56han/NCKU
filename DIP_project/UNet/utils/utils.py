import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img, cmap='gray')
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_overlap_mask(img, mask, true_masks, dice_score):
    img_np = np.array(img)
    mask_np = np.array(mask)
    true_mask_np = np.array(true_masks)
    mask_modified = mask_np.copy() 
    
    true_mask_np_3_channel = np.squeeze(true_mask_np, axis=0)
    true_mask_np_3_channel = np.repeat(true_mask_np_3_channel[:, :, np.newaxis], 3, axis=2)  
    
    mask_np_8_bit = cv2.convertScaleAbs(mask_modified)
    mask_np_3_channel = cv2.cvtColor(mask_np_8_bit, cv2.COLOR_GRAY2RGB)
    mask_np_3_channel *= 255
    true_mask_np_3_channel *= 255
    white_pixels = np.all(mask_np_3_channel == [255, 255, 255], axis=-1)  # 找出值為白色的像素位置
    true_white_pixels = np.all(true_mask_np_3_channel == [255, 255, 255], axis=-1)  # 找出值為白色的像素位置

    mask_np_3_channel[white_pixels] = [255, 0, 0] # red
    true_mask_np_3_channel[true_white_pixels] = [0, 255, 0] # green
    
    alpha = 0.75  # 調整alpha值以控制重疊的透明度
    img_np = img_np.astype(np.uint8)
    true_mask_np_3_channel = true_mask_np_3_channel.astype(np.uint8)
    pred_result = cv2.addWeighted(img_np, 1, true_mask_np_3_channel, 1 - alpha, 0)
    pred_result_np = np.array(pred_result)
    pred_result_np = pred_result_np.astype(np.uint8)
    mask_np_3_channel = mask_np_3_channel.astype(np.uint8)
    result = cv2.addWeighted(pred_result_np, 1, mask_np_3_channel, 1 - alpha, 0)

    plt.imshow(result)
    plt.title('Test Dice score: {}'.format(dice_score))
    plt.axis('off')
    plt.show()