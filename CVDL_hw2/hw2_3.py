import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np

def dimension_reduction(filename):
    image = cv2.imread(filename)

    # Normalize gray scale image from [0,255] to [0,1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0

    height, width = normalized.shape
    min_dimension = min(height, width)
    print("Minimum dimension:", min_dimension)

    # reconstruction error less or equal to 3.0
    target_error = 3.0
    components = min_dimension

    while True:
        # Use PCA to do dimension reduction from min(w,h) to n
        # find the minimum components 
        pca = PCA(n_components = components)
        transformed = pca.fit_transform(normalized) # 回傳降維資料的數據
        reconstructed = pca.inverse_transform(transformed) # 將降維後的資料轉換成原始資料
        reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)
        # reconstructed *= 255.0

        # Use MSE to compute reconstruction error
        mse = mean_squared_error(gray, reconstructed)
        if mse >= target_error:
            break

        components -= 1

    components += 1
    # Print out the n value
    print("Number of components:", components)
    print("Reconstruction error:", mse)

    plt.figure(figsize=(10, 5))

    image2rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    plt.title("logo.jpg")
    plt.imshow(image2rgb, cmap="gray")
    plt.axis("off")
    
    # Plot the gray scale image 
    plt.subplot(1, 3, 2)
    plt.title("Gray scale image")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    # the reconstruction image with n components
    plt.subplot(1, 3, 3)
    plt.title(f"Reconstructed Image\nwith {components} Components")
    plt.imshow(reconstructed, cmap="gray")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    filename = r".\Dataset_Cvdl_Hw2\Q3\logo.jpg"

    dimension_reduction(filename)
