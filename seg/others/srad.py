import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取影像（灰階模式）
image = cv2.imread('ultrasound_image.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化參數
num_iterations = 10  # 演化的迭代次數，控制 SRAD 的濾波強度，較高的值會增加計算時間，但能更顯著地減少雜訊
delta_t = 0.25       # 時間步長，通常設為 0.25，過大可能導致數值不穩定
kappa = 30           # 控制邊緣敏感性，值越小，對邊緣的保護越強，但可能降低去噪效果

def srad(image, num_iterations, delta_t, kappa):
    # 將影像轉為浮點型數據
    img = image.astype(np.float32)

    # 正規化
    img /= 255.0

    for _ in range(num_iterations):
        # 計算梯度
        gradient_x = np.diff(img, axis=1, prepend=img[:, :1])
        gradient_y = np.diff(img, axis=0, prepend=img[:1, :])

        # 計算梯度的平方
        grad_square = gradient_x**2 + gradient_y**2

        # 計算局部均值
        mean_square = cv2.blur(img**2, (3, 3))
        mean = cv2.blur(img, (3, 3))
        q0_square = mean_square - mean**2

        # 計算邊緣檢測函數
        c = np.exp(-grad_square / (kappa**2 * (q0_square + 1e-10)))

        # 計算擴散項
        div_c_grad = np.diff(c * gradient_x, axis=1, prepend=c[:, :1]) + \
                     np.diff(c * gradient_y, axis=0, prepend=c[:1, :])

        # 更新影像
        img += delta_t * div_c_grad

    return (img * 255).astype(np.uint8)

if __name__ == '__main__':
    # 執行 SRAD 演算法
    filtered_image = srad(image, num_iterations, delta_t, kappa)

    # 顯示結果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Filtered Image (SRAD)")
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')

    plt.show()
