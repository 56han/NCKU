import sys
import cv2
from PIL import Image, ImageFilter
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from mid_hw import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog,QMessageBox, QLabel
from matplotlib import pyplot as plt

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        super(MainWindow_controller, self).__init__(parent) # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.img_array = None
        self.image = None

    def setup_control(self):
        self.ui.LoadImageBtn.clicked.connect(self.load_image)
        self.ui.SmoothFilterBtn.clicked.connect(self.smooth_filter)
        self.ui.SharpBtn.clicked.connect(self.sharp)
        self.ui.GaussianBtn.clicked.connect(self.gaussian)
        self.ui.LowerpassBtn.clicked.connect(self.ft_gaussian)

    def load_image(self):
        filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./")                 # start path
        # self.ImageR = cv2.imread(filename,0) 
        print("load image right from", filename)
        self.image = Image.open(filename)
        pixmap = QPixmap(filename)
        OriginalImage = pixmap.scaled(256, 256)
        self.ui.Image_LU.setScaledContents(True)
        self.ui.Image_LU.setPixmap(OriginalImage)

        # Convert the image to a NumPy array
        self.img_array = np.array(self.image)

    def Fourier_Transform_noise(self):
        f = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(f)
        # 构建振幅谱
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        # 显示频率域图像
        # plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Frequency Domain Image')
        # plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        # plt.show()

        # 生成圓形遮罩
        rows, cols = self.image.size
        center_x, center_y = rows // 2, cols // 2  # 圓心座標
        radius = 40  # 圓的半徑
        # 生成網格
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center_x)**2 + (y - center_y)**2 <= radius**2  # 生成圓形遮罩
        # 在一個黑色背景上，將圓形區域設置為 1
        mask = np.zeros((rows, cols))
        mask[mask_area] = 1

        # mask
        fshift_filtered = fshift * mask

        spectrum = 20*np.log(np.abs(fshift_filtered)+1)
        # 显示频率域图像
        # plt.imshow(spectrum, cmap='gray')
        # plt.title('Frequency Domain Image')
        # plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        # plt.show()

        img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
        img_back = np.abs(img_back)

        # Display or save the restored images
        ft_noise_img = Image.fromarray(img_back.astype(np.uint8))
        
        ft_noise_img.save('1_ft_noise_image.jpg')
        pixmap = QPixmap('1_ft_noise_image.jpg')
        ft_Image = pixmap.scaled(256, 256)
        self.ui.Image_RD.setScaledContents(True)
        self.ui.Image_RD.setPixmap(ft_Image)
        self.ui.ImageLabel_RD.setText('1(b) Fourier Transform')

    def Fourier_Transform_sharp(self):
        f = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(f)
        # 构建振幅谱
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        # 显示频率域图像
        # plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Frequency Domain Image')
        # plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        # plt.show()

        # 生成圓形遮罩
        rows, cols = self.image.size
        center_x, center_y = rows // 2, cols // 2  # 圓心座標
        radius = 30  # 圓的半徑
        # 生成網格
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center_x)**2 + (y - center_y)**2 <= radius**2  # 生成圓形遮罩
        # 在一個黑色背景上，將圓形區域設置為 1
        mask = np.ones((rows, cols))
        mask[mask_area] = 0

        # 將高頻部分遮罩掉
        fshift_filtered = fshift * mask

        spectrum = 20*np.log(np.abs(fshift_filtered)+1)
        # 显示频率域图像
        # plt.imshow(spectrum, cmap='gray')
        # plt.title('Frequency Domain Image')
        # plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        # plt.show()

        # 逆傅立葉變換回空間域
        img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
        img_back = np.abs(img_back)


        # 將銳化後的圖像轉換為與原始圖像相同的類型
        img_back = np.uint8(img_back)

        # 將銳化後的圖像與原始圖像相加
        sharp_img = cv2.addWeighted(self.img_array, 1.0, img_back, 1, 0)

        ft_sharp_img = Image.fromarray(sharp_img.astype(np.uint8))
        
        ft_sharp_img.save('2_ft_sharp_img.jpg')
        pixmap = QPixmap('2_ft_sharp_img.jpg')
        ft_Image = pixmap.scaled(256, 256)
        self.ui.Image_RD.setScaledContents(True)
        self.ui.Image_RD.setPixmap(ft_Image)
        self.ui.ImageLabel_RD.setText('2(b) Fourier Transform')

    def smooth_filter(self): #1
        # Apply the average filter
        ksize = (5, 5) 
        smooth_image_arr = cv2.blur(self.img_array, ksize) 

        avg_image = Image.fromarray(smooth_image_arr.astype(np.uint8))

        # Show and save the resulting images
        # sharp_image.show()
        avg_image.save('1_avg_filtered_image.jpg')
        pixmap = QPixmap('1_avg_filtered_image.jpg')
        avg_Image = pixmap.scaled(256, 256)
        self.ui.Image_RU.setScaledContents(True)
        self.ui.Image_RU.setPixmap(avg_Image)
        self.ui.ImageLabel_RU.setText('1(a) Average filter')

        #--------------------------------------------------------------------------------------------------
        # Apply the median filter
        # median_filtered_img = Image.fromarray(self.img_array.copy())
        # median_filtered_img = median_filtered_img.filter(ImageFilter.MedianFilter(size=3))  # Change the size as needed
        
        medianBlur = cv2.medianBlur(self.img_array, 5)
        median_filtered_img = Image.fromarray(medianBlur.astype(np.uint8))

        # median_filtered_img.show()
        median_filtered_img.save('1_median_filtered_image.jpg')
        pixmap = QPixmap('1_median_filtered_image.jpg')
        avg_Image = pixmap.scaled(256, 256)
        self.ui.Image_LD.setScaledContents(True)
        self.ui.Image_LD.setPixmap(avg_Image)
        self.ui.ImageLabel_LD.setText('1(a) Median filter')

        self.Fourier_Transform_noise()

    def sharp(self): #2
        sobelx = cv2.Sobel(self.img_array, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.img_array, cv2.CV_64F, 0, 1, ksize=3)
        # sobel_combined = cv2.magnitude(sobelx, sobely).astype('uint8')  # 綜合 Sobel 邊緣檢測結果
        sobel_combined = cv2.magnitude(sobelx, sobely)  # 綜合 Sobel 邊緣檢測結果

        # sobel_combined = cv2.imshow("sobel_combined",sobel_combined)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 將 Sobel 圖像轉為 8 位元
        sobel_combined = np.uint8(sobel_combined)

        # 將 Sobel 圖像與原始圖像結合（加權相加）
        sobel_combined_image = cv2.addWeighted(self.img_array, 1, sobel_combined, 0.3, 0.0)

        # Show and save the resulting images
        # sharp_image.show()
        sobel_image = Image.fromarray(sobel_combined_image.astype(np.uint8))

        sobel_image.save('2_sharp_image.jpg')
        pixmap = QPixmap('2_sharp_image.jpg')
        sobel = pixmap.scaled(256, 256)
        self.ui.Image_LD.setScaledContents(True)
        self.ui.Image_LD.setPixmap(sobel)
        self.ui.ImageLabel_LD.setText('2(a) Sobel mask')
        self.ui.ImageLabel_RU.setText('No use')
        self.ui.Image_RU.setPixmap(QPixmap(""))
        
        self.Fourier_Transform_sharp()

    def gaussian(self): #3
        # Create a 5x5 Gaussian filter
        gaussian_filter = cv2.getGaussianKernel(5, 0)  # Generating a 5x1 Gaussian kernel
        gaussian_filter = np.outer(gaussian_filter, gaussian_filter.transpose())  # Creating a 5x5 Gaussian filter

        # Apply the Gaussian filter to the image using filter2D
        gaussian_img = cv2.filter2D(self.img_array, -1, gaussian_filter)

        gaussian_image = Image.fromarray(gaussian_img.astype(np.uint8))

        gaussian_image.save('3_gaussian_image.jpg')
        pixmap = QPixmap('3_gaussian_image.jpg')
        gaussian = pixmap.scaled(256, 256)
        self.ui.Image_RU.setScaledContents(True)
        self.ui.Image_RU.setPixmap(gaussian)
        self.ui.ImageLabel_RU.setText('Result')
        self.ui.ImageLabel_LD.setText('No use')
        self.ui.Image_LD.setPixmap(QPixmap(""))
        self.ui.ImageLabel_RD.setText('No use')
        self.ui.Image_RD.setPixmap(QPixmap(""))

    def ft_gaussian(self): #4
        # Apply Fourier transform to the image
        f = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(f)
        # 构建振幅谱
        # magnitude_spectrum = 20*np.log(np.abs(fshift))

        # Create a Gaussian filter in the frequency domain
        rows, cols = self.image.size
        crow, ccol = rows // 2, cols // 2
        gaussian_filter = np.zeros((rows, cols), np.float32)

        # Create Gaussian filter mask
        sigma = 30  # Adjust sigma as needed for smoothing level
        for i in range(rows):
            for j in range(cols):
                gaussian_filter[i, j] = np.exp(-((i - crow) ** 2 + (j - ccol) ** 2) / (2 * sigma ** 2))

        # Apply Gaussian filter to the Fourier transformed image
        filtered_fshift = fshift * gaussian_filter

        # Inverse Fourier transform
        f_ishift = np.fft.ifftshift(filtered_fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Display or save the restored images
        ft_gaussian_img = Image.fromarray(img_back.astype(np.uint8))
        
        ft_gaussian_img.save('4_ft_gaussian_img.jpg')
        pixmap = QPixmap('4_ft_gaussian_img.jpg')
        ft_gaussian = pixmap.scaled(256, 256)
        self.ui.Image_RU.setScaledContents(True)
        self.ui.Image_RU.setPixmap(ft_gaussian)
        self.ui.ImageLabel_RU.setText('Result')
        self.ui.ImageLabel_LD.setText('No use')
        self.ui.Image_LD.setPixmap(QPixmap(""))
        self.ui.ImageLabel_RD.setText('No use')
        self.ui.Image_RD.setPixmap(QPixmap(""))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())