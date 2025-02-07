import os
from PIL import Image
import copy
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision.transforms import transforms
from torchvision import transforms, datasets
# from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights


class MyResNet50(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        pretrain_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        pretrain_model.fc = nn.Sequential() # remove fc layer
        self.features = pretrain_model
        fc_in_channels = self.features.layer4[-1].conv3.out_channels
        self.fc = nn.Linear(fc_in_channels, num_class)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def ResNet50_inference(test_model, pil_image, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = test_model.to(device)

    inference_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64), antialias=True),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image_normalized = inference_transforms(pil_image)
    image_normalized = image_normalized.unsqueeze_(0)
    image_normalized = image_normalized.to(device)

    out = test_model(image_normalized)
    _, pred = torch.max(out, 1)
    pred_class_name = classes[int(pred)]

    return pred_class_name


if __name__ == "__main__":
    # show_images()

    # inference
    class_name = ["Cancer", "Mix", "Warthin", "Normal"]
    num_class = len(class_name)
    test_model_path = r"runs\20240106_1701\weights\ResNet50_state_dict_20240106_1701_0.8633.pth"
    test_model = MyResNet50(num_class)
    test_model.load_state_dict(torch.load(test_model_path))

    image_path = "./newdataset/test/Cancer/c283-17842193_16_L.png"
    image = Image.open(image_path).convert("RGB")

    pred_class_name = ResNet50_inference(test_model, image, class_name)

    print(pred_class_name)