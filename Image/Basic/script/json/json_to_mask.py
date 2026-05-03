import argparse
import cv2
import json
import numpy as np
import os
from tqdm import tqdm


def json_mask(args):
    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)

    json_list = np.array(os.listdir(args.json_dir))
    json_list = json_list[[jpg.endswith(".json") for jpg in json_list]]

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith(".jpg") for jpg in jpg_list]]

    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.json_dir, json_name)

        # jpg
        jpg_name = str(json_name).replace(".json", ".jpg")
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        jpg = cv2.imread(jpg_path)

        # read json
        with open(json_path, "r", encoding="UTF-8") as fr:
            try:
                annotation = json.load(fr)
            except:
                print(json_path)
                continue

        img_width = annotation["imageWidth"]
        img_height = annotation["imageHeight"]
        img_shape = [img_height, img_width]

        # mask
        mask = np.zeros(img_shape, dtype=jpg.dtype)
        mask_name = json_name.replace(".json", ".png")
        mask_path = os.path.join(args.output_dir, mask_name)

        for track in annotation["shapes"]:
            label = track["label"]
            points = np.array(track["points"])

            # contour
            contour = points
            contour = [np.array(contour).reshape(-1, 1, 2)]

            mask = cv2.drawContours(mask, contour, -1, (255, 255, 255), cv2.FILLED)
        cv2.imwrite(mask_path, mask)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/yuanhuan/data/image/Liang_Fang/original/Irregular_goods/202410/sample_data/880_1010/",
    )
    args = parser.parse_args()

    args.jpg_dir = os.path.join(
        args.input_dir,
        "JPEGImages",
    )
    args.json_dir = os.path.join(
        args.input_dir,
        "Jsons",
    )
    args.output_dir = os.path.join(
        args.input_dir,
        "Masks",
    )

    json_mask(args)
