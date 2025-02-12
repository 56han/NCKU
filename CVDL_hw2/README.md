# Computer Vision and Deep Learning - Homework 2

This project is part of the Computer Vision and Deep Learning (CVDL) coursework. It consists of five main tasks: Background Subtraction, Optical Flow, PCA for Dimension Reduction, MNIST Classification using VGG19, and Cat-Dog Classification using ResNet50. A PyQt5-based UI is used to present all tasks.

## 1. Background Subtraction  
### Given:  
- Traffic video: `traffic.mp4`  

### Task:  
- Remove background and show the result.  

### Steps:  
1. Load the video using File Dialog.  
2. Create a background subtractor using: `cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows=True)`
 
3. For each frame in the video:
    - Apply Gaussian blur: `cv2.GaussianBlur(frame, (5, 5), 0)`
    - Get background mask using `subtractor.apply()`.
    - Extract moving objects using `cv2.bitwise_and()`.

## 2. Optical Flow
### Given: 
- Video: `optical_flow.mp4`

### 2.1 Preprocessing
- Click button `2.1` to detect the point at the bottom of the doll's nose in the first frame.
- Use `cv2.goodFeaturesToTrack()` to detect the point.
- Mark the detected point with a red cross using `cv2.line()`.

### 2.2 Video Tracking
- Click button `2.2` to track the detected point throughout the video using `cv2.calcOpticalFlowPyrLK()`.
- Display the trajectory of the tracking point using `cv2.line()`, using a highly visible color (e.g., yellow `(0,100,255)`).

## 3. PCA - Dimension Reduction
### Given: 
- RGB image: `logo.jpg`

### Task:
- Use Principal Component Analysis (PCA) for dimension reduction.
- Find the minimum number of components `n` that keeps the reconstruction error ≤ 3.0.

### Steps:
1. Convert RGB image to grayscale.
2. Normalize pixel values from [0,255] to [0,1].
3. Perform PCA and reconstruct the image.
4. Compute reconstruction error using Mean Squared Error (MSE).
5. Find the minimum `n` where `error ≤ 3.0` and print `n`.
6. Plot the original and reconstructed images.

## 4. Training a MNIST Classifier Using VGG19 with BN
### 4.1 Load Model and Show Model Structure
- Click button `1. Show Model Structure` to display the VGG19 with Batch Normalization (BN) model using: `torchsummary.summary()`

### 4.2 Show Training/Validation Accuracy and Loss
- Train the model for at least 30 epochs.
- Record training/validation accuracy and loss for each epoch.
- If validation accuracy is low, try:
    - Adjusting the learning rate.
    - Modifying data augmentation techniques.
- Save the model with the highest validation accuracy.
- Plot training and validation accuracy/loss curves using `matplotlib.pyplot.plot()`.

### 4.3 Run Inference
- Load the trained model with the highest validation accuracy.
- Draw a number on a graffiti board (black background, white pen).
- Click button `3. Predict` to run inference.
- Display the predicted class label on the GUI.
- Show the probability distribution of predictions in a histogram.
- Click button `4. Reset` to clear the board.

## 5. Train a Cat-Dog Classifier Using ResNet50
### 5.1 Load the Dataset and Resize Images
- Load inference dataset.
- Resize images to `224×224×3 (RGB)`.
- Click button `1. Show Images` to display one image from each class.

### 5.2 Plot Class Distribution of Training Dataset
- Build a ResNet50 model with a Sigmoid activation function.
- Replace the output layer with a fully connected (FC) layer of 1 node.
- Show model structure in the terminal.

### 5.3 Show ResNet50 Model Structure
- Train and validate two ResNet50 models.
- Plot accuracy values using a bar chart and save the figure.

### 5.4 Improve ResNet50 with Random-Erasing
- Load the trained model.
- Click button `Load Image` to select an image.
- Resize the image to `224×224×3 (RGB)`.
- Click button `5. Inference` to classify the image.
- Display the predicted class label.

### 5.5 Use the Trained Model for Inference
- Load a trained ResNet50 model.
- Show the predicted class label for an input image.

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