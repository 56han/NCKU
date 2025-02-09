from torchvision import datasets, transforms
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np 
def show_dataset():
    inference_dataset_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        #    transforms.ToTensor(),                                
                                        #    torchvision.transforms.Normalize(
                                        #        mean=[0.485, 0.456, 0.406],
                                        #        std=[0.229, 0.224, 0.225],),
                                        ])
    dir = r'.\Dataset_Cvdl_Hw2_Q5\dataset\inference_dataset'
    inference_dataset = datasets.ImageFolder(dir, transform=inference_dataset_transforms)

    cat_image = None
    dog_image = None
    for (img, label) in inference_dataset:
        if label == 0:
            if cat_image == None:
                cat_image = img
        elif label == 1:
            if dog_image == None:
                dog_image = img
        if cat_image != None and dog_image != None:
            break

    cat_image_array = np.array(cat_image)
    dog_image_array = np.array(dog_image)

    plt.subplot(121)
    plt.imshow(cat_image_array)
    plt.axis("off")
    plt.title("cat")
    plt.subplot(122)
    plt.imshow(dog_image_array)
    plt.axis("off")
    plt.title("dog")
    plt.show()
