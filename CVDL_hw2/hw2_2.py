import cv2
import numpy as np

'''
goodFeaturesToTrack():偵測到的角點仍是像素等級
不僅支援 Harris 角點偵測
也支援 Shi Tomasi Algo 角點偵測
cornerSubPix():更精細的角點座標，亞像素
'''

def preprocessing(filename):
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect the point at the bottom of the doll's nose
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7
    )
    x, y = map(int, corners[0, 0])
    # show the point with a red cross mark 繪製角點
    cv2.line(frame, (x - 30, y), (x + 30, y), (0, 0, 255), 5)
    cv2.line(frame, (x, y - 30), (x, y + 30), (0, 0, 255), 5)

    width = 800
    height = int(frame.shape[0] * (width / frame.shape[1]))
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", width, height)

    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()


def video_tracking(filename):
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    ret, frame1 = cap.read()
    old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(
        old_gray, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7
    )

    mask = np.zeros_like(frame1)
    while True:
        ret, frame = cap.read()
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print("No frame captured or frame is empty")

        # Track the point on the whole video using OpenCV function 
        pl, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
        )

        good_old = p0[st == 1]
        good_new = pl[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            # Display the trajectory of the tracking point throughout the video using cv2.line
            # Pick a highly visible color (0, 100, 255)
            mask = cv2.line(mask, (a, b), (c, d), (0, 100, 255), 2)
            cv2.line(frame, (a - 30, b), (a + 30, b), (0, 0, 255), 5)
            cv2.line(frame, (a, b - 30), (a, b + 30), (0, 0, 255), 5)

        img = cv2.add(frame, mask)

        width = 800
        height = int(frame.shape[0] * (width / frame.shape[1]))
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", width, height)

        cv2.imshow("Result", img)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    filename = r".\Dataset_Cvdl_Hw2\Q2\optical_flow.mp4"

    preprocessing(filename)
    video_tracking(filename)
