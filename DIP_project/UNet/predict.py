import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from utils.utils import plot_img_and_mask, plot_overlap_mask
import cv2
import torch.nn as nn
import torch
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import matplotlib.pyplot as plt

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


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # in_files = args.input
    in_files = [r'..\output_mask_test\images\b58-15991078_11_0.png ',
                r'..\output_mask_test\images\b58-15991078_11_1.png',
                r'..\output_mask_test\images\b59-15762303_9_0.png',
                r'..\output_mask_test\images\b59-15762303_9_1.png',
                r'..\output_mask_test\images\b60-02986412_8_0.png',
                r'..\output_mask_test\images\b60-02986412_8_1.png',
                r'..\output_mask_test\images\b61-08688114_6_0.png',
                r'..\output_mask_test\images\b61-08688114_6_1.png',
                r'..\output_mask_test\images\b70-09071962_9_0.png',
                r'..\output_mask_test\images\b70-09071962_9_1.png',
                r'..\output_mask_test\images\b72-04955055_3_0.png',
                r'..\output_mask_test\images\b72-04955055_3_1.png',
                r'..\output_mask_test\images\c230-11996869_12_0.png',
                r'..\output_mask_test\images\c230-11996869_12_1.png',
                r'..\output_mask_test\images\c281-17880678_9_0.png',
                r'..\output_mask_test\images\c281-17880678_9_1.png',
                r'..\output_mask_test\images\c283-17842193_16_0.png',
                r'..\output_mask_test\images\c283-17842193_16_1.png']
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    true_masks_folder = r'output_mask_test\masks'
    dice_score_list = []

    for i, filename in enumerate(in_files):
        dice_score = 0
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        
        name = os.path.basename(filename)
        mask_file_path = os.path.join("..",true_masks_folder, name)
        true_masks = Image.open(mask_file_path)
        transform = transforms.ToTensor()
        pre_mask = transform(mask)
        true_masks = transform(true_masks)
        
        # true_masks = torch.as_tensor(true_masks.copy()).long().contiguous()
        criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        with torch.autocast(device.type, enabled=True):
            if net.n_classes == 1:
                assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
                pre_mask = (F.sigmoid(pre_mask) > 0.5).float()
                # compute the Dice score
                # dice_score += dice_coeff(mask_pred, true_masks, reduce_batch_first=False)
                dice_score += dice_coeff(pre_mask.squeeze(1), true_masks, reduce_batch_first=False) # dice_score : tensor type
            else:
                assert true_masks.min() >= 0 and true_masks.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                true_masks = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                pre_mask = F.one_hot(pre_mask.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(pre_mask[:, 1:], true_masks[:, 1:], reduce_batch_first=False)

            logging.info('Test Dice score: {}'.format(dice_score))
        
        dice_score_list.append(dice_score.float().item())
        plot_overlap_mask(img, mask, true_masks, dice_score)
        
        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
            # cv2.imwrite('2.png', mask)
    
    print(dice_score_list)
        
