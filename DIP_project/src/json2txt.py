import os
import json
import numpy as np
import glob
import argparse

def json2txt(json_dir):
    """
    將json檔案轉換成txt檔案, class: left normal=0, right normal=1
    """
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    print(f"{json_dir}: {len(json_files)} annotations")
    for file in json_files:
        bonding_box = []
        json_name = os.path.basename(file)   #get filename(remove path)
        file_name = os.path.splitext(json_name)[0] #get filename(remove.json)
        txt_path = os.path.join(json_dir, file_name+ '.txt')
        #open json file
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # get class, x_center, y_center, width, height
        class_name = None
        for i in range(len(data['shapes'])):
            if data['shapes'][i]['label'] == 'left normal':
                class_name = '0'
            elif data['shapes'][i]['label'] == 'right normal':
                class_name = '1'
            else :  # only need to detect left/right normal
                continue

            points = data['shapes'][i]['points']
            points = np.array(points)
            xs = points[:, 0]
            ys = points[:, 1]
            x_max = xs.max()
            x_min = xs.min()
            y_max = ys.max()
            y_min = ys.min()
            x_center = (x_max + x_min) / 2 / data['imageWidth']
            y_center = (y_max + y_min) / 2 / data['imageHeight']
            width = (x_max - x_min) / data['imageWidth'] 
            height = (y_max - y_min) / data['imageHeight']
            # Yolo format : class x_center y_center width height   
            bonding_box.append([class_name," ", x_center, " ", y_center, " ", width, " ", height])
            # write txt file
            with open (txt_path, 'w') as f: 
                for i in range(len(bonding_box)):
                    for j in range(len(bonding_box[i])):
                        f.write(str(bonding_box[i][j]))
                    f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str)
    args = parser.parse_args()
    json2txt(args.json_dir)