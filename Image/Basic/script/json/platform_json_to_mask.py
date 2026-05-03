import argparse
import cv2
import json
import numpy as np
import os
from tqdm import tqdm


def get_color(idx):
    idx = idx * 5
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def json_mask(args):
    # mkdir
    os.makedirs(args.output_mask_dir, exist_ok=True)
    os.makedirs(args.output_mask_jpg_dir, exist_ok=True)

    json_list = np.array(os.listdir(args.json_dir))
    json_list = json_list[[jpg.endswith(".json") for jpg in json_list]]

    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.json_dir, json_name)

        # jpg
        jpg_name = str(json_name).replace(".json", ".jpg")
        jpg_path = os.path.join(args.jpg_dir, jpg_name)

        # mask
        mask_name = json_name.replace(".json", ".png")
        output_mask_path = os.path.join(args.output_mask_dir, mask_name)
        output_mask_img_path = os.path.join(args.output_mask_jpg_dir, mask_name)

        # read json
        with open(json_path, "r", encoding="UTF-8") as fr:
            try:
                annotation = json.load(fr)
            except:
                print(json_path)
                continue
        
        # mask
        img = cv2.imread(jpg_path)
        try:
            mask = np.zeros(img.shape, dtype=img.dtype)
        except:
            print(jpg_path)
            continue

        if "shapes" not in annotation:
            print(jpg_path)
            continue

        for track in annotation["shapes"]:

            if track["type"] == "polygon":

                # contour
                contour = np.array(track["points"])
                contour = [np.array(contour).reshape(-1, 1, 2)]
                mask = cv2.drawContours(mask, contour, -1, (255, 255, 255), cv2.FILLED)

            elif track["type"] == "rectangle": 
                points = np.array(track["points"])
                contour = []
                contour.append([points[0], points[1]])
                contour.append([points[2], points[1]])
                contour.append([points[2], points[3]])
                contour.append([points[0], points[3]])
                contour = [np.array(contour).reshape(-1, 1, 2)]
                mask = cv2.drawContours(mask, contour, -1, (255, 255, 255), cv2.FILLED)

        # mask_img
        mask_img = (mask != 0).astype(np.uint8)
        mask_img[:, :, 0] = mask_img[:, :, 0] * get_color(1)[0]
        mask_img[:, :, 1] = mask_img[:, :, 1] * get_color(1)[1]
        mask_img[:, :, 2] = mask_img[:, :, 2] * get_color(1)[2]
        mask_img = cv2.addWeighted(
            src1=img, alpha=0.8, src2=mask_img, beta=0.5, gamma=0.0
        )

        cv2.imwrite(output_mask_path, mask)
        cv2.imwrite(output_mask_img_path, mask_img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/yuanhuan/data/image/Liang_Fang/training/Irregular_goods_20241101/46_0906_1/",
    )
    args = parser.parse_args()

    args.jpg_dir = os.path.join(args.input_dir, "JPEGImages")
    args.json_dir = os.path.join(args.input_dir, "Jsons")
    args.output_mask_dir = os.path.join(args.input_dir, "Masks")
    args.output_mask_jpg_dir = os.path.join(args.input_dir, "Mask_Imgs")

    json_mask(args)
