# Computer Vision and Deep Learning - Homework 1

This project is part of the Computer Vision and Deep Learning (CVDL) coursework. It consists of five main tasks: Camera Calibration, Augmented Reality, Stereo Disparity Map, SIFT, and Training a CIFAR10 Classifier using VGG19 with Batch Normalization. A PyQt5-based UI is used to present all tasks.

## 1. Camera Calibration  
### 1.1 Corner Detection  
- Detect corners in the chessboard images.  

### 1.2 Find the Intrinsic Matrix  
- Compute the camera's intrinsic parameters.  

### 1.3 Find the Extrinsic Matrix  
- Compute the camera's extrinsic parameters for different views.  

### 1.4 Find the Distortion Matrix  
- Compute the distortion coefficients of the camera.  

### 1.5 Show the Undistorted Result  
- Use the computed parameters to undistort the images.

## 2. Augmented Reality  
### 2.1 Show Words on Board  
- Calibrate 5 images to obtain intrinsic, distortion, and extrinsic parameters.  
- Input an English word (≤6 characters) in the text box.  
- Use the provided library `alphabet_lib_onboard.txt` to render the word on the board.

### 2.2 Show Words Vertically  
- Calibrate 5 images to obtain intrinsic, distortion, and extrinsic parameters.  
- Input the same word as in Q2.1.  
- Use the provided library `alphabet_lib_vertical.txt` to render the word vertically.

## 3. Stereo Disparity Map  
### 3.1 Compute Stereo Disparity Map  
- Generate a disparity map from left and right stereo images.  

### 3.2 Checking Disparity Value  
- Click on a point in the left image to find the corresponding point in the right image.

## 4. SIFT (Scale-Invariant Feature Transform)  
### 4.1 Keypoints Detection  
- Detect keypoints in `Left.jpg` using the SIFT algorithm.  
- Draw keypoints on the left image.

### 4.2 Matched Keypoints  
- Extract good matches using SIFT.  
- Draw matched feature points between two images.

## 5. Training a CIFAR10 Classifier Using VGG19 with Batch Normalization  
### 5.1 Load CIFAR10 and Show Augmented Images  
- Load the CIFAR10 dataset and display 9 augmented images with labels.  
- Train a VGG19 model with Batch Normalization to classify 10 classes.

### 5.2 Load Model and Show Model Structure  
- VGG19: A 19-layer deep convolutional neural network.  
- BN (Batch Normalization): Speeds up training and improves stability.

### 5.3 Show Training/Validation Accuracy and Loss  
- Dataset consists of 60,000 32x32 color images in 10 classes:  
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Training set: 50,000 images  
- Validation set: 10,000 images  
- Testing set: 10 images (generated from validation set)

### 5.4 Run Inference Using Best Model  
- Train VGG model with batch normalization using PyTorch.  
- Run inference using the model with the highest validation accuracy.  
- Display predicted class distribution and label.

## UI Implementation  
- A PyQt5-based UI integrates all five tasks.  
- Users can interactively perform each task and visualize the results.

## Dependencies  
- Python 3.8 
- OpenCV  
- Matplotlib  
- PyQt5  
- PyTorch  
- Torchvision
- Torchsummary
- Tensorboard
- Pillow 

## How to Run 
1. Install required dependencies

2. Run the PyQt5 UI
    ``` sh
    python controller.py
    ```

## Author
Yi-Han Wan