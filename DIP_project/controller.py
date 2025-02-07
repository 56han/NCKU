import sys
sys.path.append('./ObjectDetection/yolov7')
import os
import torch
import cv2
from glob import glob
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from UI.final_project_UI import Ui_MainWindow # Modify UI as your ui_file.py name, Modify class name(Ui_MainWindow、Ui_Form...)
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from src.yolo_detect import mydetect
from src.json2txt import json2txt
from src.bbox import plot_one_box, calculat_evaluation_metric, label_to_xyxy
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from UNet.predict import predict_img
# from UNet.unet import UNet
# import logging
from src.mysegmentation import mypredict
from src.myResNet50 import ResNet50_inference, MyResNet50

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow() # modify class name
        self.ui.setupUi(self)
        self.move(100, 200)  # 設定視窗位置
        self.image_viewer_original = None 
        self.image_viewer_predict = None
        self.image_viewer_groundtruth = None
        self.images_folder = None
        self.images_path_list = []
        self.images_path_index = 0
        self.image_path = None
        self.image_name = None
        self.classification_model_path = "./weights/ResNet50_state_dict_20240106_1701_0.8633.pth"
        self.classification_model = MyResNet50(4)
        self.classification_model.load_state_dict(torch.load(self.classification_model_path))
        self.setup_control()

        # objectdetection 結果


    def setup_control(self):
        self.ui.pb_load_folder.clicked.connect(self.load_image_folder)
        self.ui.pb_pre_image.clicked.connect(self.pre_image)
        self.ui.pb_next_image.clicked.connect(self.next_image)
        self.ui.pb_detection.clicked.connect(self.detect)
        self.ui.pb_segmentation.clicked.connect(self.segment)

    def load_image_folder(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        print(dir)
        if dir == "":
            print("No directory selected.")
            return
        else:
            self.images_folder = dir
            print(f"Load folder: {self.images_folder}")
            self.images_path_index = 0 # 歸零 index
            self.images_path_list = glob(os.path.join(self.images_folder, "*.png"))
            self.image_viewer_original = ImageViewer('Original', self.image_path, 550, 200)
            self.set_current_image()
            # generate txt
            reply = QMessageBox.question(self, "提示" ,"要產生.txt檔嗎?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                json2txt(self.images_folder)


    def set_current_image(self):
        self.image_path = self.images_path_list[self.images_path_index]
        self.image_name = os.path.basename(self.image_path)
        self.ui.label_cuurrent_image.setText(f"Current Image : {self.image_name}")
        pixmap = QPixmap(self.image_path).scaled(500, 500)
        self.image_viewer_original.image_label.setPixmap(pixmap)
        self.image_viewer_original.show()
        # close ImageViewer
        for ImageViewer in [self.image_viewer_predict, self.image_viewer_groundtruth]:
            if ImageViewer != None:
                ImageViewer.close()
        
        # reset label
        self.ui.label_iou.setText("IoU : ")
        self.ui.label_accuracy.setText("Accuracy : ")
        self.ui.label_precision.setText("Precision : ")
        self.ui.label_recall.setText("Recall : ")
        

    def pre_image(self):
        if self.images_path_index > 0:
            self.images_path_index -= 1
            self.set_current_image()

    def next_image(self):
        if self.images_path_index < (len(self.images_path_list) - 1):
            self.images_path_index += 1
            self.set_current_image()


    def check_image_load(self):
        if not self.images_folder:
            QMessageBox.about(self, "check", "No image, Please load image first!")
            return False
        return True

    def detect(self):
        if not self.check_image_load():
            return

        parameter_dict = {
            'weights': './ObjectDetection/yolov7/weight/yolo_best.pt',
            'source': f'{self.image_path}',
            'img_size': 640,
            'conf_thres': 0.6,
            'iou_thres': 0.1,
            'device': '0' if torch.cuda.is_available() else 'cpu',\
            'view_img': False,
            # 'view-img': False,
            'save_txt': True,
            'save_conf': False,
            'trace': True,
            'nosave': False,
            'classes': None,
            'agnostic_nms': True,
            'augment': False,
            # 'project': 'runs/detect',
            'project': './ObjectDetection/detect',
            'name': 'exp',
            'exist_ok': True,
        }

        predict_txt_path = os.path.join("./ObjectDetection/detect/exp/labels", f"{self.image_name.split('.')[0]}.txt")
        if os.path.exists(predict_txt_path):
            os.remove(predict_txt_path)

        mydetect(**parameter_dict)
        self.show_predict()
        self.show_GT()
        
        gt_txt_path = f"{self.image_path.split('.')[0]}.txt"
        IoU, accuracy, precision, recall = calculat_evaluation_metric(predict_txt_path, gt_txt_path)
        self.ui.label_iou.setText(f"IoU : {IoU:.4f}")
        self.ui.label_accuracy.setText(f"Accuracy : {accuracy:.4f}")
        self.ui.label_precision.setText(f"Precision : {precision:.4f}")
        self.ui.label_recall.setText(f"Recall : {recall:.4f}")


    def show_predict(self):
        predict_image_path = os.path.join("./ObjectDetection/detect/exp", self.image_name)
        self.image_viewer_predict = ImageViewer("Predict", predict_image_path, 1100, 200) # 暫時先顯示original image
        print(predict_image_path)
        self.image_viewer_predict.show()

    def show_segment(self):
        image_filename = self.image_name.replace('.png', '')
        predict_image_path = os.path.join("./output_crop_img/predict_output", image_filename +'_1.png') # segment
        self.image_viewer_predict = ImageViewer("Predict", predict_image_path, 1100, 200) 
        print(predict_image_path)
        self.image_viewer_predict.show()

    def show_GT(self):
        # bbox colors
        class_label = ['left normal', 'right normal']
        class_color = [(255, 255, 0), (255, 0, 255)]

        txt_path = self.image_path.split(".")[0] + ".txt"
        with open(txt_path, "r") as f:
            lines = f.readlines()
            img = cv2.imread(self.image_path)
            for line in lines:
                class_index, x, y, w, h = line.split(" ")
                class_index, x, y, w, h = int(class_index), float(x), float(y), float(w), float(h)
                img_size = 512
                x = [(x-(w/2))*img_size, (y-(h/2))*img_size, (x+w/2)*img_size, (y+h/2)*img_size]
                plot_one_box(x, img, class_color[class_index], class_label[class_index])

        ground_truth_dir = "./ObjectDetection/GroundTruth"
        if not os.path.exists(ground_truth_dir):
            os.mkdir(ground_truth_dir)
            print(f"create dir {ground_truth_dir}")

        gt_path = os.path.join(ground_truth_dir, f"{os.path.splitext(self.image_name)[0]}_GT.png")
        cv2.imwrite(gt_path, img)
        self.image_viewer_groundtruth = ImageViewer("Ground Truth", gt_path, 1650, 200)
        self.image_viewer_groundtruth.show()

    def show_seg_GT(self):
        image_filename = self.image_name.replace('.png', '')
        com_image_path = os.path.join("./segmentation/GT", image_filename +'_com.png') # segment
        # com_img = Image.open(com_image_path)
        self.image_viewer_groundtruth = ImageViewer("Ground Truth", com_image_path, 1650, 200)
        self.image_viewer_groundtruth.show()
    
    def closeEvent(self, event):
        """
        關閉主視窗時, 關閉所有的image viewer
        """
        for child_window in [self.image_viewer_original, self.image_viewer_predict, self.image_viewer_groundtruth]:
            if child_window != None:
                child_window.close()

    def get_crop_img(self,txt_path):
        with open(txt_path, 'r') as file:
            content = file.readlines()
        img = cv2.imread(self.image_path)
        self.image_name = os.path.basename(self.image_path)
        image_filename = self.image_name.replace('.png', '')
        img_size = 512
        output_folder_img = 'output_crop_img/images'
        os.makedirs(output_folder_img, exist_ok=True)
        for line in content:
            data = line.split()  # 按空格分割每行數據
            class_index, x_center, y_center, width, height = data

            if class_index == '0':
                class_index, left_min_x, left_min_y, left_max_x, left_max_y = label_to_xyxy(data, img_size)
                left_x = left_max_x - left_min_x
                left_y = left_max_y - left_min_y
                cropped_img_L = img[math.floor(left_min_y):math.ceil(left_min_y + left_y), math.floor(left_min_x):math.ceil(left_min_x + left_x)]
                L_img_np = np.array(cropped_img_L)
                output_img_path_L = os.path.join(output_folder_img, image_filename + '_0.png')
                Image.fromarray(L_img_np).save(output_img_path_L)
            elif class_index == '1':
                class_index, right_min_x, right_min_y, right_max_x, right_max_y = label_to_xyxy(data, img_size)
                right_x = right_max_x - right_min_x
                right_y = right_max_y - right_min_y
                cropped_img_R = img[math.floor(right_min_y):math.ceil(right_min_y + right_y), math.floor(right_min_x):math.ceil(right_min_x + right_x)]
                R_img_np = np.array(cropped_img_R)
                output_img_path_R = os.path.join(output_folder_img, image_filename + '_1.png')
                Image.fromarray(R_img_np).save(output_img_path_R)
        return output_img_path_L, output_img_path_R
            
    def classification(self, pil_image):
        class_name = ["Cancer", "Mix", "Warthin", "Normal"]
        pred_class_name = ResNet50_inference(self.classification_model, pil_image, class_name)

        return pred_class_name
        
    def segment(self):
        if not self.check_image_load():
            return
        
        # step 1: crop
        predict_txt_path = os.path.join("./ObjectDetection/detect/exp/labels", f"{self.image_name.split('.')[0]}.txt")
        if not os.path.exists(predict_txt_path):
            QtWidgets.QMessageBox.warning(self, "Warning", "Please detect first.")
            return

        left_img_path, right_img_path = self.get_crop_img(predict_txt_path)

        # classfication 
        left_img = Image.open(left_img_path)
        right_img = Image.open(right_img_path)
        left_class = self.classification(left_img)
        right_class = self.classification(right_img)
        
        # step 2: predict + overlap to origin img
        output_mask = 'output_crop_img/predict_output'
        os.makedirs(output_mask, exist_ok=True)
        input_list = []
        input_list.append(left_img_path)
        input_list.append(right_img_path)
        parameter_dict = {
            'model' : 'UNet/checkpoints/checkpoint_epoch62.pth',
            'input' : input_list,
            'output' : f'{output_mask}',
            'viz': True,
            'no_save':True,
            'mask_threshold': 0.5,
            'scale': 0.5,
            'bilinear': False,
            'classes': 1,
            'original_img_path': self.image_path,
            'left_class' : left_class,
            'right_class' : right_class,
            'path': self.image_path
        }
        dice_score = mypredict(**parameter_dict)

        # step 3: show
        self.show_segment()
        self.show_seg_GT()
        self.ui.label_dice_coefficient.setText(f"Dice : {dice_score*100:.3f} %")



class ImageViewer(QtWidgets.QDialog):
    """
    用來顯示圖片的 ImageViewer
    """
    def __init__(self, title, image_path, x, y):
        super().__init__()

        self.setWindowTitle(title)
        self.setGeometry(0, 0, 350, 350)
        self.move(x, y)
        self.image_label = QtWidgets.QLabel()

        # size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.image_label.setSizePolicy(size_policy)
        # self.image_label.setGeometry(0, 0, 350, 350)
        # self.image_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(image_path).scaled(500, 500)
        print(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)

        self.setLayout(layout)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())