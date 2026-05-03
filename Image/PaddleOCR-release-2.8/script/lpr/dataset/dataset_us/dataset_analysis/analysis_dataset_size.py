import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm


def analysis_dataset_size(args):

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    # 初始化一个字典来存储宽度和高度的分布
    size_distribution = {'width': {}, 'height': {}}

    for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        plate_roi = row['roi']
        plate_roi_list = plate_roi.split(',')
        x1 = int(plate_roi_list[0])
        x2 = int(plate_roi_list[1])
        y1 = int(plate_roi_list[2])
        y2 = int(plate_roi_list[3])

        width = x2 - x1
        height = y2 - y1

        # 更新尺寸分布字典
        size_distribution['width'].setdefault(width, 0)
        size_distribution['width'][width] += 1
        
        size_distribution['height'].setdefault(height, 0)
        size_distribution['height'][height] += 1

    # 绘制尺寸分布图
    plt.figure(figsize=(12, 6))

    # 绘制宽度分布
    plt.subplot(1, 2, 1)
    width_distribution_sorted = sorted(size_distribution['width'].items())
    plt.bar([label for label, count in width_distribution_sorted], [count for label, count in width_distribution_sorted])
    plt.title('Width Distribution')

    # 绘制高度分布
    plt.subplot(1, 2, 2)
    height_distribution_sorted = sorted(size_distribution['height'].items())
    plt.bar([label for label, count in height_distribution_sorted], [count for label, count in height_distribution_sorted])
    plt.title('Height Distribution')

    plt.savefig(os.path.join(args.input_dir, 'size_distribution.png'), dpi=300)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Alabama_2022") 
    parser.add_argument('--ocr_name', type=str, default="plate_us_202406") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")  
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.ocr_name, args.date_name)
    args.input_csv_path = os.path.join(args.input_dir, '{}.csv'.format(args.date_name))

    analysis_dataset_size(args)