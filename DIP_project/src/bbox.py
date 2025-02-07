import cv2
import random
import numpy as np

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# 0: backgound
# 1: predict
# 2: grouund truth
# 3: intersection(TP)

def calculat_evaluation_metric(predict_txt, gt_txt):
    image_size = 512
    with open(predict_txt) as predict_f:
        predict_lines = predict_f.readlines()
    with open(gt_txt) as gt_f:
        gt_lines = gt_f.readlines()
    
    array = np.zeros((image_size, image_size))

    for line in predict_lines:
        predict_box = line.split(" ")
        class_index, x_min, y_min, x_max, y_max = label_to_xyxy(predict_box, image_size)
        for x in range(x_min-1, x_max):
            for y in range(y_min-1, y_max):
                array[x][y] += 1 # 1: predict

    for line in gt_lines:
        gt_box = line.split(" ")
        class_index, x_min, y_min, x_max, y_max = label_to_xyxy(gt_box, image_size)
        for x in range(x_min-1, x_max):
            for y in range(y_min-1, y_max):
                array[x][y] += 2 # 2: grouund truth

    # 計算特定值的出現次數
    count_1 = np.sum(array == 1)
    count_2 = np.sum(array == 2)
    count_3 = np.sum(array == 3)

    TP = count_3
    FP = count_1
    TN = image_size*image_size - (count_1 + count_2 + count_3)
    FN = count_2

    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"TN: {TN}")
    print(f"FN: {FN}")

    IoU = TP / (count_1 + count_2 + count_3)
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)


    return [IoU, accuracy, precision, recall]


# label format: [x_center, y_center, width, height] (normalized)
# bbox format: [x_min, y_min, x_max, y_max]
# box1: predict box
# box2: ground truth box

def label_to_xyxy(label, image_size):
    class_index, x_center, y_center, width, height = label
    x_min = int((float(x_center) - float(width) / 2)*image_size)
    y_min = int((float(y_center) - float(height) / 2)*image_size)
    x_max = int((float(x_center) + float(width) / 2)*image_size)
    y_max = int((float(y_center) + float(height) / 2)*image_size)

    return [class_index, x_min, y_min, x_max, y_max]


# def calculate_TP(predict_box, gt_box):
#     # 計算兩個矩形框的交集區域
#     x_left = max(predict_box[0], gt_box[0])
#     y_top = max(predict_box[1], gt_box[1])
#     x_right = min(predict_box[2], gt_box[2])
#     y_bottom = min(predict_box[3], gt_box[3])

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0  # 沒有交集
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     return intersection_area


# def calculate_FP(predict_box, gt_box):
#     # x1: predict
#     x1_left = predict_box[0]
#     y1_top = predict_box[1]
#     x1_right = predict_box[2]
#     y1_bottom = predict_box[3]
#     # x2: ground truth
#     x2_left = gt_box[0]
#     y2_top = gt_box[1]
#     x2_right = gt_box[2]
#     y2_bottom = gt_box[3]

#     TP = calculate_TP(predict_box, gt_box)
#     predict_area = (x1_right - x1_left) * (y1_bottom - y1_top)
#     if TP > 0:
#         FP = predict_area - TP
#     else:
#         FP = predict_area

#     return FP

# def calculate_TN(predict_box, gt_box):

#     pass


# def calculate_FN(predict_box, gt_box):
#     # x1: predict
#     x1_left = predict_box[0]
#     y1_top = predict_box[1]
#     x1_right = predict_box[2]
#     y1_bottom = predict_box[3]
#     # x2: ground truth
#     x2_left = gt_box[0]
#     y2_top = gt_box[1]
#     x2_right = gt_box[2]
#     y2_bottom = gt_box[3]

#     TP = calculate_TP(predict_box, gt_box)
#     gt_area = (x2_right - x2_left) * (y2_bottom - y2_top)

#     if TP > 0:
#         FN = gt_area - TP
#     else:
#         FN = gt_area

#     return FN


# def calculate_iou(box1, box2):
#     # 計算兩個矩形框的交集區域
#     x_left = max(box1[0], box2[0])
#     y_top = max(box1[1], box2[1])
#     x_right = min(box1[2], box2[2])
#     y_bottom = min(box1[3], box2[3])

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0  # 沒有交集

#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     # 計算兩個矩形框的聯集區域
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - intersection_area

#     iou = intersection_area / union_area
#     return iou