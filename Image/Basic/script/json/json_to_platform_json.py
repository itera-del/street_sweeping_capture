import argparse
import json
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, "/yuanhuan/code/demo/Image")
from Basic.utils.folder_tools import *
from Basic.script.json.platform_json_write import PlatformJsonWriter


def json_2_platform_json(args):

    # mkdir
    create_folder(args.platform_json_dir)

    json_list = np.array(os.listdir(args.json_dir))
    json_list = json_list[[json.endswith(".json") for json in json_list]]
    json_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith(".jpg") for jpg in jpg_list]]

    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.json_dir, json_name)

        # jpg
        jpg_name = str(json_name).replace(".json", ".jpg")
        assert jpg_name in jpg_list

        # read json
        with open(json_path, "r", encoding="UTF-8") as fr:
            annotation = json.load(fr)

        img_width = annotation["imageWidth"]
        img_height = annotation["imageHeight"]

        json_bboxes = {}
        for track in annotation["shapes"]:
            label = track["label"]
            points = track["points"]
            type = track["shape_type"]
            assert type == "polygon"

            points_list = []
            for idy in range(len(points)):
                point_idx = points[idy]
                points_list.append(point_idx[0])
                points_list.append(point_idx[1])

            if label not in json_bboxes:
                json_bboxes[label] = []
            json_bboxes[label].append(points_list)

        out_json_path = os.path.join(args.platform_json_dir, json_name)
        platform_json_writer.write_json(
            img_width,
            img_height,
            jpg_name,
            out_json_path,
            frame_num=idx,
            polygon_dict=json_bboxes,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/yuanhuan/data/image/Liang_Fang/original/Irregular_goods/202410/sample_data/880_1010/",
    )
    args = parser.parse_args()

    # safeisland
    args.jpg_dir = os.path.join(
        args.input_dir,
        "JPEGImages",
    )
    args.json_dir = os.path.join(
        args.input_dir,
        "Jsons_LabelImg",
    )
    args.platform_json_dir = os.path.join(
        args.input_dir,
        "Jsons",
    )
    json_2_platform_json(args)
