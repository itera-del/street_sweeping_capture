import argparse
import multiprocessing as mp
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.json.platform_json_write import PlatformJsonWriter


def xml_2_platform_json(info:Dict):

    fid = info.pop("fid")
    total_dirs = info.pop("total_dirs")
    src_imgpath = info.pop("src_imgpath")
    src_xmlpath = info.pop("src_xmlpath")
    dst_jsonpath = info.pop("dst_jsonpath")
    gid = info.pop("gid")
    total_groups = info.pop("total_groups")
    args = info.pop("args")

    if gid % 1000 == 0:
        print(f"processing file={fid}/{total_dirs},\tgroup={gid}/{total_groups}")
    
    # 如果图像文件不存在
    if not os.path.isfile(dst_jsonpath):

        try:
            jpg_name = os.path.basename(src_imgpath)

            # read xml
            tree = ET.parse(src_xmlpath)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
            root = tree.getroot()   # 获取根节点

            img_width = int(root.find('size').find('width').text)
            img_height = int(root.find('size').find('height').text)
            # print(img_width, img_height)

            # 标签检测和标签转换
            rect_list = []
            for object in root.findall('object'):
                # name
                classname = str(object.find('name').text)

                # change name
                if classname in args.change_label_dict.keys():
                    classname = args.change_label_dict[classname]
                if classname not in args.label_list:
                    print("Ignore: {}".format(classname))
                    continue
                assert classname in args.label_list 

                # bbox
                bbox = object.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox.find(pt).text))
                    bndbox.append(cur_pt)
                
                bndbox.extend([classname])
                rect_list.append(bndbox)
            
            platform_json_writer = PlatformJsonWriter()
            platform_json_writer.write_json(img_width, img_height, jpg_name, dst_jsonpath, frame_num=gid, rect_list=rect_list)
        except:
            print("Failed to xml_2_platform_json: ", src_imgpath)
            return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", "-d", type=str, default="/yuanhuan/data/image/RM_Capture/training/Capture_Plate_90w_dupes_0_95_512_288", help="数据根目录")
    parser.add_argument("--outdir", "-o", type=str, default="/yuanhuan/data/image/RM_Capture/training/Capture_Plate_90w_dupes_0_95_512_288", help="数据输出目录")
    parser.add_argument("--imgdir-name", "-in", type=str, default="JPEGImages", help="图像文件夹名字")
    parser.add_argument("--annodir-name", "-an", type=str, default="Annotations", help="标注文件夹名字")
    parser.add_argument("--out-annodir-name", "-outan", type=str, default="Json", help="标注文件夹名字")
    parser.add_argument("--nproc", "-n", type=int, default=8, help="进程数")
    args = parser.parse_args()

    # Capture
    args.task_name = "Capture"
    args.label_list = ['car', 'bus', 'truck', 'motorcyclist', 'license']
    args.change_label_dict = {'license_plate': 'license'}

    img_indir = os.path.join(args.data_roots, args.imgdir_name)
    anno_indir = os.path.join(args.data_roots, args.annodir_name)
    anno_outdir = os.path.join(args.outdir, args.out_annodir_name)
    os.makedirs(anno_outdir, exist_ok=True)

    # 获取所有的图像
    items = []
    for filename in os.listdir(img_indir):
        file_basename = filename[:-4]
        suffix = filename[-3:]
        if suffix in ["jpg","png","jpeg","bmp"]:
            imgpath = os.path.join(img_indir, filename)
            annopath = os.path.join(anno_indir, f"{file_basename}.xml")
            out_annopath = os.path.join(anno_outdir, f"{file_basename}.json")
            items.append((imgpath, annopath, out_annopath))
    print(f"Found {len(items)} images")


    # 将图像重新组织一下, 变成多进程的输入
    grouped_items = []
    for i, item in enumerate(items, start=1):
        # 增加一些的信息
        item_info = {}
        item_info["fid"] = 1
        item_info["total_dirs"] = 1
        item_info["src_imgpath"] = item[0]
        item_info["src_xmlpath"] = item[1]
        item_info["dst_jsonpath"] = item[2]
        item_info["gid"] = i
        item_info["total_groups"] = len(items)
        item_info["args"] = args
        grouped_items.append(item_info)

    # for idx in range(len(grouped_items)):
    #     xml_2_platform_json(grouped_items[idx])

    # 进程池
    MAX_PROCESS = min(len(grouped_items), args.nproc)
    pool = mp.Pool(MAX_PROCESS)

    # 多进程处理
    final_infos = []
    pool.map(xml_2_platform_json, grouped_items)