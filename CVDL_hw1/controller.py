import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap
import tkinter as tk
from tkinter import filedialog
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog,QMessageBox, QLabel
from UI import Ui_MainWindow
import os
import cv2 
import time
import numpy as np
from matplotlib import pyplot as plt
import hw1_main
import copy
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable


class CustomVGG19(nn.Module):
    def __init__(self):
        super(CustomVGG19, self).__init__()
        pretrain_model = models.vgg19_bn(pretrained=True)
        pretrain_model.classifier = nn.Sequential()  # remove last layer
        self.features = pretrain_model

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)  # 10 classes for classification
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        super(MainWindow_controller, self).__init__(parent) # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.file_path = ""
        self.imageR=None
        self.imageL=None
        self.frameSize = (2048,2048)
        self.size = (11,8)
        self.input_text = []
        self.file_1 = None
        self.file_2 = None
        self.folder_path = ""

    def setup_control(self):
        # TODO
        self.ui.lineEdit.setText('Happyy')
        self.ui.Load_Folder_btn.clicked.connect(self.load_file)
        self.ui.Load_Image_L_btn.clicked.connect(self.load_image_L)
        self.ui.Load_Image_R_btn.clicked.connect(self.load_image_R)

        self.ui.Find_Corners_btn.clicked.connect(self.Find_Corners)        #1.1
        self.ui.Find_Istrinsic_btn.clicked.connect(self.Find_Istrinsic)    #1.2
        self.ui.Find_Extrinsic_btn.clicked.connect(self.Find_Extrinsic)    #1.3
        self.ui.Find_Distortion_btn.clicked.connect(self.Find_Distortion)  #1.4
        self.ui.Show_Result_btn.clicked.connect(self.Show_Result)  #1.5
         
        self.ui.Show_Word_on_Bound_btn.clicked.connect(self.Show_Word_on_Bound)  #2.1
        self.ui.Show_Words_Vertically_btn.clicked.connect(self.Show_Words_Vertically)  #2.2

        self.ui.pushButton.clicked.connect(self.Stereo_Disparity_Map) #3.1

        self.ui.load_image_1_btn.clicked.connect(self.load_image_1) #4.1
        self.ui.Load_image_2_btn.clicked.connect(self.load_image_2)
        self.ui.keypoint_btn.clicked.connect(self.show_keypoint)
        self.ui.matched_keypoint_btn.clicked.connect(self.matched_keypoint)

        self.ui.load_image_VGG19_btn.clicked.connect(self.load_image_VGG) 
        self.ui.Show_Augmented_Images_btn.clicked.connect(self.Show_Augmented_Images) #5.1
        self.ui.Show_Model_Structure_btn.clicked.connect(self.Show_Model_Structure) #5.2
        self.ui.Show_Accuracy_Loss_btn.clicked.connect(self.Show_Accuracy_Loss) #5.3
        self.ui.Inference_btn.clicked.connect(self.Inference) #5.4

    
      
    def load_file(self):   #叫出選擇檔案視窗
        self.file_path = QFileDialog.getExistingDirectory(None, "QFileDialog.getExistingDirectory()","./Dataset_CvDl_Hw1/")
        print(self.file_path)
    def load_image_R(self):
        filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./Dataset_CvDl_Hw1/")                 # start path
        self.ImageR = cv2.imread(filename,0) 
        print("load image right from", filename)
    def load_image_L(self):
        filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./Dataset_CvDl_Hw1/")                 # start path
        self.ImageL = cv2.imread(filename,0) 
        print("load image left from", filename)

  # 1
    def Find_Corners(self):
        try :
            print("Find_Corners")
            self.ui.Find_Corners_btn.setDisabled(True)
            hw1_main.find_chessboard(self.file_path)
            self.ui.Find_Corners_btn.setDisabled(False)
        except:
            print("NO Find_Corners")
            QMessageBox.about(self, "檢查輸入", "Please Load folder first")
            self.ui.Find_Corners_btn.setDisabled(False)
    
    def Find_Istrinsic(self):
        try:
            hw1_main.Find_The_Intrinsic_Matrix(self.file_path)
        except:
            QMessageBox.about(self, "檢查輸入", "Please Load folder first")

    def Find_Extrinsic(self):
        try:
            select_number = self.ui.comboBox.currentText()  #1.3
            # print(select_number)
            hw1_main.Find_The_Extrinsic_Matrix(self.file_path,select_number)
        except:
            QMessageBox.about(self, "檢查輸入", "Please Load folder first")

    def Find_Distortion(self):
        try:
            hw1_main.Find_The_Distortion_Matrix(self.file_path)
        except:
            QMessageBox.about(self, "檢查輸入", "Please Load folder first")

    def Show_Result(self):
        try:
            hw1_main.Show_the_undistorted_result(self.file_path)
        except:
            QMessageBox.about(self, "檢查輸入", "Please Load folder first")

    def get_input_text(self):
        self.input_text = self.ui.lineEdit.text()
        self.input_text = self.input_text.replace(' ', '')
        if len(self.input_text) > 6 or len(self.input_text) < 0:
            QMessageBox.about(self, "檢查輸入", "Please enter word and less than 6.")
        
    # 2.1   
    def Show_Word_on_Bound(self):
        self.get_input_text()
        try:
            hw1_main.Show_Word_on_Image(self.file_path,self.input_text,"onboard")
        except:
            QMessageBox.about(self, "檢查輸入", "Please Load folder first")
        
    # 2.2
    def Show_Words_Vertically(self):
        self.get_input_text()
        try:
            hw1_main.Show_Word_on_Image(self.file_path,self.input_text,"vertical")
        except:
            QMessageBox.about(self, "檢查輸入", "Please Load folder first")
    # 3.1
    def Stereo_Disparity_Map(self):
        
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(self.ImageL,self.ImageR)

        dis = disparity.astype(np.float64)
        
        disparity = cv2.normalize(dis, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        
        def draw_point(event):
            d = disparity[int(event.ydata)][int(event.xdata)]
            x = event.xdata - d
            y = event.ydata
            circle2 = plt.Circle((x, y), 15, color='lime')
            
            if d!=0:
                ax2.clear()
                print('('+ str(int(x)) + ','+ str(int(y)) +')' +',dis: ' + str(d) + '\n')
                ax2.add_patch(circle2)
                imgR = cv2.cvtColor(self.ImageR, cv2.COLOR_GRAY2RGB)
                plt.axis('off')
                ax2.imshow(imgR)
                figR.canvas.draw()
            else:
                print("Failure case")

        figL = plt.figure(figsize=(12, 8)) 
        imgL = cv2.cvtColor(self.ImageL, cv2.COLOR_GRAY2RGB)
        ax1 = figL.add_subplot(111)
        figL.canvas.manager.set_window_title("img L")
        plt.axis('off')
        plt.imshow(imgL)
        
        figR = plt.figure(figsize=(12, 8))
        imgR = cv2.cvtColor(self.ImageR, cv2.COLOR_GRAY2RGB)
        ax2 = figR.add_subplot(111)
        figR.canvas.manager.set_window_title("img R")
        plt.axis('off')
        plt.imshow(imgR)
        
        figG = plt.figure(figsize=(12, 8))
        figG.canvas.manager.set_window_title("Disparity img")
        plt.axis('off')
        plt.imshow(disparity, 'gray')

        cid = figL.canvas.mpl_connect('button_press_event', draw_point)
        
        plt.show()

    # 4.1
    def load_just_file(self):   #叫出選擇檔案視窗
        file_path, filetype = QFileDialog.getOpenFileName(self,"Open file","./Dataset_CvDl_Hw1/")
        return file_path
    
    def load_image_1(self):
        self.file_1 = self.load_just_file()
        print("load image 1 from : ",self.file_1)
        
    def load_image_2(self):
        self.file_2 = self.load_just_file()
        print("load image 2 from : ",self.file_2)
    
    def show_keypoint(self):
        img = cv2.imread(self.file_1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #建立sift物件        
        sift = cv2.xfeatures2d.SIFT_create()
        #進行檢測和計算  返回特徵點資訊和描述
        keypoints , descriptor = sift.detectAndCompute(gray, None)
        #keypoints：特徵點集合list，向量內每一個元素是一個KeyPoint物件，包含了特徵點的各種屬性資訊；
    
        img = cv2.drawKeypoints(img, keypoints,img, color=(0,255,0))
        
        img = cv2.resize(img, (512,512))
        cv2.imshow('Keypoints',img)
        cv2.moveWindow("Keypoints", 255, 255) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 4.2
    def matched_keypoint(self):
        print("matched keypoint")
        img = cv2.imread(self.file_1)
        car = cv2.imread(self.file_2)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        car_gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
        #建立sift物件        
        sift = cv2.xfeatures2d.SIFT_create()
        #進行檢測和計算  返回特徵點資訊和描述
        img_keypoints , img_descriptor = sift.detectAndCompute(img_gray, None)
        car_keypoints , car_descriptor = sift.detectAndCompute(car_gray, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(img_descriptor,car_descriptor, k=2)

        # Extract Good Matches
        goodMatch = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatch.append(m)

        goodMatch = np.expand_dims(goodMatch, 1)
        # Draw the matched feature points between two image
        img_out = cv2.drawMatchesKnn(img_gray, img_keypoints, car_gray, car_keypoints, goodMatch, None, matchesMask=None,
                               singlePointColor=(0, 255, 0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img_out = cv2.resize(img_out, (2048,1024))
        cv2.imshow('matched keypoint', img_out)
        cv2.moveWindow("matched keypoint", 255, 255) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 5.1
    def Show_Augmented_Images(self):
        folder_path = QFileDialog.getExistingDirectory(None, "QFileDialog.getExistingDirectory()","./Dataset_CvDl_Hw1/Q5_image")

        # List all files in the folder
        image_files = os.listdir(folder_path)

        a1 = transforms.RandomHorizontalFlip(p=0.5) # p=0.5: 一半的圖片會被水平翻轉
        a2 = transforms.RandomVerticalFlip(p=0.5)
        a3 = transforms.RandomRotation(30)

        # Create a list to store the loaded and transformed images
        trans_images = []
        labels = []

        # Sort the image files for consistent order
        image_files.sort()

        # Load and apply the transformation to each image
        for image_file in image_files[:9]:  # Load the first 9 images
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)  # Load the image using PIL
            image = a1(image)  # Apply the RandomHorizontalFlip transformation
            image = a2(image)
            image = a3(image)
            trans_images.append(image)
            # Extract the filename (without extension)
            label = os.path.splitext(image_file)[0]
            labels.append(label)

        rows = 3
        cols = 3
        fig = plt.figure(figsize=(10, 10))

        for i in range(9):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(trans_images[i])  # Display the ith image
            title = os.path.basename(labels[i])
            plt.title(title)
            plt.axis('off')  # Turn off axis labels

        plt.tight_layout()  # Adjust subplot parameters for better layout
        plt.show()  # Display the images

    #5.2    
    def Show_Model_Structure(self):
        # Create the VGG19 model with batch normalization
        custom_vgg19 = CustomVGG19()

        # Check if a GPU is available and move the model and input data to it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        custom_vgg19 = custom_vgg19.to(device)

        # Display the model structure using torchsummary.summary
        summary(custom_vgg19, (3, 32, 32))

    #5.3
    def Show_Accuracy_Loss(self):
        img1 = cv2.imread("./vgg19_Loss.jpg") # input img
        img2 = cv2.imread("./vgg19_Accuracy.jpg") # input img
        # 確保兩張照片具有相同的尺寸
        if img1.shape[:2] != img2.shape[:2]:
            raise ValueError("兩張照片的尺寸不同")

        # 將兩張照片垂直合併
        merged_image = np.vstack((img1, img2))
        # 顯示合併後的照片
        cv2.imshow('Show Accuracy Loss', merged_image)
        cv2.moveWindow('Show Accuracy Loss', 255, 255) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #5.4
    def load_image_VGG(self):
        # global VGG_filename
        self.VGG_filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./Dataset_CvDl_Hw1/") # start path
        self.ImageVGG = Image.open(self.VGG_filename) 
        print("load image from", self.VGG_filename)
        pixmap = QPixmap(self.VGG_filename)
        scaledPixmap = pixmap.scaled(128, 128)
        #Image scaling: Use the label's setScaledContents(True) method to adapt the label size
        self.ui.label_Q5_4_img.setScaledContents(True)
        print(scaledPixmap.height())
        print(scaledPixmap.width())
        self.ui.label_Q5_4_img.setPixmap(scaledPixmap)
    
    def Inference(self):
        classes = ['plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        model = CustomVGG19()
        model = torch.load("./VGG19_cifar10.pth")
        print("load model successfully")

        model.eval()  # Set the model to evaluation mode
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std) 
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = transform(self.ImageVGG)
        img = img.unsqueeze(0)  # Add a batch dimension
        img = img.to(device).detach()
        
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', classes[predicted.item()])
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(outputs.cpu())   # probabilities[0] is the probability of each class


        probs = probabilities[0].detach().numpy()
        plt.figure(figsize=(5, 5))
        plt.title("Probability of each class")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.bar(classes, probs)
        plt.show()
        self.ui.label_Q5_predict.setText(f'Predicted: {classes[predicted.item()]}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())
    