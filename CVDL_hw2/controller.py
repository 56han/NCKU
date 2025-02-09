from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel
from PyQt5.QtGui import QMouseEvent, QPaintEvent, QPainter, QColor, QPixmap, QPen, QImage
from PyQt5.QtCore import QBuffer
from hw2_ui import Ui_MainWindow
# import hw2_ui as ui
from hw2_1 import bg_subtraction
from hw2_2 import preprocessing, video_tracking
from hw2_3 import dimension_reduction
from CustomDataset import show_dataset

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import cv2 
import numpy as np
from PIL import Image
import io
from torchvision import transforms
import torchvision
from matplotlib import pyplot as plt
import glob

class CustomVGG19(nn.Module):
    def __init__(self):
        super(CustomVGG19, self).__init__()
        pretrain_model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
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

class Main(QtWidgets.QMainWindow):
    def __init__(self):
         super().__init__()
        #  self.ui = ui()
         self.ui = Ui_MainWindow()
         self.ui.setupUi(self)
         self.setup_control()
         self.video_path = None
         self.image_path = None
         self.img = None

    def setup_control(self):
        self.ui.load_image.clicked.connect(self.load_image)
        self.ui.load_video.clicked.connect(self.load_video)

        self.ui.bg_subtraction.clicked.connect(self.Background_Subtraction)
        self.ui.preprocessing.clicked.connect(self.preprocessing)
        self.ui.video_tracking.clicked.connect(self.v_tracking)
        self.ui.dim_reduction.clicked.connect(self.dim_reduction)

        self.lastPoint = QtCore.QPoint()
        self.endPoint = QtCore.QPoint()
        self.drawing = False
        self.image = QtGui.QImage(self.ui.label_GraffitiWall.size(), QtGui.QImage.Format_RGB32)
        # Set the initial background color
        self.image.fill(QtCore.Qt.black)
        # Connect the reset button to the clear_drawing function
        self.ui.reset.clicked.connect(self.clear_drawing)
        # Connect the label_GraffitiWall to mouse events
        self.ui.label_GraffitiWall.mousePressEvent = self.mouse_press_event
        self.ui.label_GraffitiWall.mouseMoveEvent = self.mouse_move_event
        self.ui.label_GraffitiWall.mouseReleaseEvent = self.mouse_release_event
        self.ui.label_GraffitiWall.paintEvent = self.paint_event

        self.ui.show_model1.clicked.connect(self.Show_Model_VGG19Structure)
        self.ui.show_acc_loss.clicked.connect(self.Show_Acc_Loss)
        self.ui.predict.clicked.connect(self.Predict)


        self.ui.load_image2.clicked.connect(self.load_image2)
        self.ui.show_image.clicked.connect(self.show_image)
        self.ui.show_model2.clicked.connect(self.Show_Model_ResNetStructure)
        self.ui.comprasion.clicked.connect(self.show_comparison)
        self.ui.inference.clicked.connect(self.ResNet_inference)



    def load_video(self):
        file_path, filetype = QFileDialog.getOpenFileName(self,"Open file","./Dataset_Cvdl_Hw2")
        self.video_path = file_path

    def load_image(self):
        file_path, filetype = QFileDialog.getOpenFileName(self,"Open file","./hw1_q4/Q4_images")
        self.image_path = file_path

    def Background_Subtraction(self):        
        print("1.1")
        video_path = self.video_path
        try:
            bg_subtraction(video_path)
        except:
            print("load video first")

    def preprocessing(self):
        print("2.1")
        # video_path = r'Dataset_CvDl_Hw2\Q2_Image\optical_flow.mp4'
        video_path = self.video_path
        try:
            preprocessing(video_path)
        except:
            print("load video first")

    def v_tracking(self):
        print("2.2")
        # video_path = r'Dataset_CvDl_Hw2\Q2_Image\optical_flow.mp4'    
        video_path = self.video_path
        try:
            video_tracking(video_path)
        except:
            print("load video first")

    def dim_reduction(self):
        print("3")
        image_path = self.image_path
        try:
            dimension_reduction(image_path)
        except:
            print("load image first")

    # Q4
    def clear_drawing(self):
        self.image.fill(QtCore.Qt.black)  # Clear the drawing by filling with black
        self.ui.label_GraffitiWall.update()

    def mouse_press_event(self, event: QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.drawing = True

    def mouse_move_event(self, event: QMouseEvent):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.drawing:
            self.endPoint = event.pos()
            self.draw_line()

    def mouse_release_event(self, event: QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton and self.drawing:
            self.endPoint = event.pos()
            self.draw_line()
            self.drawing = False

    def draw_line(self):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QPen(QtCore.Qt.white, 5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, self.endPoint)
        self.lastPoint = self.endPoint
        self.ui.label_GraffitiWall.update()

    def paint_event(self, event: QPaintEvent):
        painter = QtGui.QPainter(self.ui.label_GraffitiWall)
        painter.drawImage(event.rect(), self.image, event.rect())

    def Show_Model_VGG19Structure(self):
        print("4.1")
        custom_vgg19 = CustomVGG19()

        # Check if a GPU is available and move the model and input data to it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        custom_vgg19 = custom_vgg19.to(device)

        # Create a random input tensor and move it to the same device
        # input_data = torch.randn(1, 3, 224, 224).to(device)

        # Display the model structure using torchsummary.summary
        summary(custom_vgg19, (3, 32, 32))

    def Show_Acc_Loss(self):
        print("4.2")
        img1 = cv2.imread("./vgg19_Loss.jpg") # input img
        img2 = cv2.imread("./vgg19_Accuracy.jpg") # input img
        # 確保兩張照片具有相同的尺寸
        if img1.shape[:2] != img2.shape[:2]:
            raise ValueError("兩張照片的尺寸不同")

        # 將兩張照片垂直合併
        merged_image = np.vstack((img1, img2))

        filename = 'VGG_acc_loss_merge.jpg'
        cv2.imwrite(filename, merged_image)
        cv2.imshow('Show Accuracy Loss', merged_image)
        cv2.moveWindow('Show Accuracy Loss', 255, 255) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
        # print("load image from", filename)
        # pixmap = QPixmap(filename)
        # scaledPixmap = pixmap.scaled(361, 151)
        # #图像缩放：使用label的setScaledContents(True)方法，自适应label大小
        # self.ui.label_GraffitiWall.setScaledContents(True)
        # print(scaledPixmap.height())
        # print(scaledPixmap.width())
        # self.ui.label_GraffitiWall.setPixmap(scaledPixmap)

    def qimage_to_pli(self, img):
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        img.save(buffer, "PNG")
        pil_im = Image.open(io.BytesIO(buffer.data()))
        return pil_im
    
    def Predict(self):
        img = self.qimage_to_pli(self.image)
        self.Inference(img)

    def Inference(self, img):
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        model = CustomVGG19()
        model = torch.load("./VGG19_MNIST.pth")
        # custom_vgg19.load_state_dict(checkpoint)
        print("load model successfully")

        model.eval()  # Set the model to evaluation mode
        mean, std = [0.5], [0.5]
        transform = transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

        # 將PIL圖像轉換為Tensor
        img_tensor = transform(img)

        # 將圖像轉移到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

        print('Predict = ', classes[predicted.item()])
        self.ui.Q4_predict.setText(f'Predict = {classes[predicted.item()]}')
        # Convert outputs to probabilities
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(outputs.cpu())    #probabilities[0] is the probability of each class

        probs = probabilities[0].detach().numpy()
        plt.figure(figsize=(5, 5))
        plt.title("Probability of each class")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.bar(classes, probs)
        plt.show()
        
    def show_image(self):
        print("5.1 show image")
        show_dataset()

    def Show_Model_ResNetStructure(self):
        print("5.2")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        model = model.to('cuda')
        # input_data = input_data.to('cuda')
        summary(model, (3, 224, 224))

    def show_comparison(self):
        print("5.3")
        without_rand = 98.94 
        with_rand = 99.06
        y = [without_rand, with_rand]
        labels = ['Without Random Erasing', 'With Random Erasing']
        plt.bar( labels, y, color='blue')
        plt.ylabel('Accuracy(%)')
        plt.title('Accuracy comparison')
        plt.text(x=0,y=without_rand,s=f"{without_rand}")
        plt.text(x=1,y=with_rand,s=f"{with_rand}")
        plt.ylim(0, 100)  
        plt.savefig("Accuracy Comparison figure")
        plt.show()

    def load_image2(self):
        file_path, filetype = QFileDialog.getOpenFileName(self,"Open file","./Dataset_Cvdl_Hw2_Q5/dataset/inference_dataset")
        self.img = cv2.imread(file_path) 
        pixmap = QPixmap(file_path)
        scaledPixmap = pixmap.scaled(150, 150)
        #图像缩放：使用label的setScaledContents(True)方法，自适应label大小
        self.ui.Q5_load_img.setScaledContents(True)
        print(scaledPixmap.height())
        print(scaledPixmap.width())
        self.ui.Q5_load_img.setPixmap(scaledPixmap)

    def ResNet_inference(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            #Define the model architecture
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
            # Load the trained model weights
            model_path = r".\best_resnet50_model.pth"
            #model.load_state_dict(torch.load(model_path))
            #Load the Weights Partially
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path)

            # Exclude keys that don't match
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # Update the model weights
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # Move the model to GPU
            model.to(device)
            #Put the model in evaluation mode
            model.eval()
            #Prepare input data
            test_image = self.img
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            test_image = cv2.resize(test_image,(224,224))
            test_image = test_image.astype('float32')/255.0 
            test_image = np.expand_dims(test_image, axis=0)
            test_image = np.transpose(test_image, (0, 3, 1, 2))
            input_tensor  = torch.tensor(test_image)
            input_tensor = input_tensor.to(device)
            print(f"Input Tensor Shape: {test_image.shape}")
            # print(model)
            predictions = model(input_tensor)
            # predictions = model.predict(test_image)
            predict = predictions[0][0]
            result_label = ""
            if predict < 0.5:
                result_label = 'Cat'
            else:
                result_label = 'Dog'
            print(result_label)
            result_text = "Predict = " + result_label
            self.ui.Q5_predict.setText(result_text)

        
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())