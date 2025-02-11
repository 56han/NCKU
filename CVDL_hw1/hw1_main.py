print("hi")
from dataclasses import dataclass
import cv2 
import numpy as np
import pathlib
import tkinter as tk
from tkinter import filedialog
import os
from numpy.core.fromnumeric import resize


class myCamera():
    def __init__(self, image_path, chessboardSize):
        self.size = chessboardSize
        self.image = cv2.imread(image_path)
        self.frameSize = (2048,2048)
        self.intrinsic_matrix = 0
        self.dist = 0 # distortion coefficients
        self.rvecs = 0 # rotation vectors
        self.tvecs = 0 # translation vectors
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 停止優化的標準
        self.corners2 = None
        self.dataPreprocessing()

    def dataPreprocessing(self):
        objp = np.zeros((self.size[0]*self.size[1],3), np.float32) # 11 * 8 * 3
        objp[:,:2] = np.mgrid[0:self.size[0],0:self.size[1]].T.reshape(-1,2)
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        # Arrays to store object points and image points from all the images.
        ret, corners = cv2.findChessboardCorners(gray, self.size, None) # 使用函数的默认设置来执行角点检测
    
        if ret == True: # 成功检测到了棋盘格
            self.objpoints.append(objp)
            #self.objpoints = np.array(self.objpoints)
            self.corners2 = cv2.cornerSubPix(gray, corners,(5,5),(-1,-1), self.criteria)
            self.imgpoints.append(self.corners2)

            # self.imgpoints = np.array(self.imgpoints)
            # print(len(self.imgpoints[0]))
  
        ret, self.intrinsic_matrix, self.disSt, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.frameSize, None, None)

    

 # 黑白格有(12,7)，內部點數是(11,8)
def find_chessboard(file_path, size=(11,8)):
    for i in range(1,len(os.listdir(file_path))+1):
        img_path = file_path + "/{}.bmp".format(i) # .format() 會填入{}
        cmr = myCamera(img_path, size)

        cv2.drawChessboardCorners(cmr.image,patternSize=(size), corners=(cmr.imgpoints[0])  ,patternWasFound=(True)) 
        show_window_name = "1.1 find chessboard of image {}".format(i)
        #show = cv2.resize(cmr.image,(800,800))
        cv2.namedWindow(show_window_name,0)          # 為了要進行圖片的縮放，所以要先建立一個視窗
        cv2.resizeWindow(show_window_name, 1000, 1000) # 對圖片進行縮小 ,方便查看
        cv2.imshow(show_window_name, cmr.image)    # 顯示結果圖片
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        

def Find_The_Intrinsic_Matrix(file_path, size=(11,8)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 停止優化的標準

    objp = np.zeros((8*11,3), np.float32) # 11 * 8 * 3
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i in range(1,len(os.listdir(file_path))+1):
        img_path = file_path + "/{}.bmp".format(i)

        image = cv2.imread(img_path)

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Arrays to store object points and image points from all the images.
        ret, corners = cv2.findChessboardCorners(gray, size, None) # 使用函数的默认设置来执行角点检测
    
        if ret == True: # 成功检测到了棋盘格
            objpoints.append(objp)
            #self.objpoints = np.array(self.objpoints)
            corners2 = cv2.cornerSubPix(gray, corners,(5,5),(-1,-1), criteria)
            imgpoints.append(corners2)
            
    ret, intrinsic_matrix, disSt, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
    print('-------------------------------------------------')
    print('1.2 The Intrinsic Matrix of Image_{}:'.format(i))
    print(intrinsic_matrix)


def Find_The_Extrinsic_Matrix(file_path, select_image, size=(11,8)):
    print(select_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 停止優化的標準

    objp = np.zeros((8*11,3), np.float32) # 11 * 8 * 3
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i in range(1,len(os.listdir(file_path))+1):
        img_path = file_path + "/{}.bmp".format(i)
        # cmr = myCamera(img_path, size)

        image = cv2.imread(img_path)

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Arrays to store object points and image points from all the images.
        ret, corners = cv2.findChessboardCorners(gray, size, None) # 使用函数的默认设置来执行角点检测
    
        if ret == True: # 成功检测到了棋盘格
            objpoints.append(objp)
            #self.objpoints = np.array(self.objpoints)
            corners2 = cv2.cornerSubPix(gray, corners,(5,5),(-1,-1), criteria)
            imgpoints.append(corners2)
            
    ret, intrinsic_matrix, disSt, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    R = cv2.Rodrigues(rvecs[int(select_image)-1])
    print('-----------------------------------------------------')
    print('1.3 The Extrinsic Matrix of selected Image_{}:'.format(select_image))
    Extrinsic_Matrix = np.hstack((R[0], tvecs[int(select_image)-1]))
    print(Extrinsic_Matrix)

def Find_The_Distortion_Matrix(file_path, size=(11,8)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 停止優化的標準

    objp = np.zeros((8*11,3), np.float32) # 11 * 8 * 3
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i in range(1,len(os.listdir(file_path))+1):
        img_path = file_path + "/{}.bmp".format(i)

        image = cv2.imread(img_path)

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Arrays to store object points and image points from all the images.
        ret, corners = cv2.findChessboardCorners(gray, size, None) # 使用函数的默认设置来执行角点检测
    
        if ret == True: # 成功檢測到了棋盤格
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners,(5,5),(-1,-1), criteria)
            imgpoints.append(corners2)
            
    ret, intrinsic_matrix, disSt, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('--------------------------------------------------')
    print('1.4 The Distortion Matrix ')
    print(disSt)

def Show_the_undistorted_result(file_path, size=(11,8)):

    for i in range(1,len(os.listdir(file_path))+1):
        img_path = file_path + "/{}.bmp".format(i)
        cmr = myCamera(img_path, size)
        h, w = cmr.image.shape[:2] 
        
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cmr.intrinsic_matrix, cmr.dist, (w,h), 1, (w,h))
        
        dst = cv2.undistort(cmr.image, cmr.intrinsic_matrix, cmr.dist, None, newCameraMatrix)
        

        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w] 
        Undistort = cv2.resize(dst,(2048,2048))
        distort = cmr.image
        merge = np.concatenate((distort,Undistort),axis=1)
        show = cv2.resize(merge,(1200,600))
        show_window_name  = 'Distorted img {} and   Undistorted img {}'.format(i,i)
        cv2.namedWindow(show_window_name, 0)
        cv2.resizeWindow(show_window_name, 1200, 600)
        cv2.imshow(show_window_name, show)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

def Show_Word_on_Image(file_path, text, type, size=(11,8)):
    text = text.upper()  
    if type == "onboard":
        fs = cv2.FileStorage(file_path +'/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)  #讀取對應的資料
    if type == "vertical":
        fs = cv2.FileStorage(file_path +'/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)  #讀取對應的資料
    
    word_arr = []     #儲存對應後的字母資料
    for j in text:
        ch = fs.getNode(j).mat()
        word_arr.append(ch)

    file_fit = []      #去掉非圖片的檔案
    for f in os.listdir(file_path):
        if f[-3:] == "bmp":
            file_fit.append(f)
    file_fit.sort()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 停止優化的標準
    objp = np.zeros((8*11,3), np.float32) # 11 * 8 * 3
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i in file_fit:
        img_path = file_path + "/" + i
        x = [[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]  #字母顯示的位置

        image = cv2.imread(img_path)

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Arrays to store object points and image points from all the images.
        ret, corners = cv2.findChessboardCorners(gray, size, None) 
    
        if ret == True: 
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners,(5,5),(-1,-1), criteria)
            imgpoints.append(corners2)
            
        ret, intrinsic_matrix, disSt, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    

    for i in range(len(file_fit)):
        img_path = file_path + "/{}.bmp".format(i+1)
        image = cv2.imread(img_path)
        for j in range(len(word_arr)):       #把字母畫在圖片上
            line = np.float32(word_arr[j] + x[j]).reshape(-1, 3)
            imgpts, jac = cv2.projectPoints(line, rvecs[i], tvecs[i], intrinsic_matrix, disSt)
            image = draw(image, imgpts)
        image = cv2.resize(image,  (1024, 1024))
        show_window_name = "Show \"{}\" on {} of image {}".format(text,type, i)
        cv2.imshow(show_window_name, image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

def draw(img, imgpts):
        imgpts = imgpts.astype(int)
        for i in range(0,len(imgpts),2):
            img = cv2.line(img,tuple(imgpts[i].ravel()),tuple(imgpts[i+1].ravel()), (255,0,0), 5)
        return img