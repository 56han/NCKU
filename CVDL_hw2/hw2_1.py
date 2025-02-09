import cv2
import numpy as np


def bg_subtraction(filename):
    # Create subtractor 創造一個背景分割器 
    # detectShadows = True 檢測陰影
    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)
    # 創造 videoCapture 對象
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    start_frame = 5
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)

    for frame_number in range(start_frame, end_frame):
        ret, frame = cap.read() #讀取影片

        if not ret:
            print(f"Error: Could not read frame {frame_number}.")
            break
        # Blur frame
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        # Get background mask 背景分割
        fg_mask = backSub.apply(blur)

        print(frame.dtype, frame.shape)
        print(fg_mask.dtype, fg_mask.shape)
        # Generate Frame 逐位元與運算
        result = cv2.bitwise_and(frame, frame, mask=fg_mask)

        cv2.imshow("rgb frame", frame)
        cv2.imshow("Foreground mask", fg_mask) # 顯示分割結果
        cv2.imshow("Result", result)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load Video from File Dialog
    filename = r".\Dataset_Cvdl_Hw2\Q1\traffic.mp4"

    bg_subtraction(filename)
