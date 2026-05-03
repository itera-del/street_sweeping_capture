# from precheck import get_data_id_list
from datasplit import *
from platform2coco import *
import mmcv
from PIL import Image
import os.path as osp
import xml.etree.ElementTree as ET
# 备忘
# 原数据标签
det_all = ['car', 'bus', 'truck', 'car_reg', 'car_big_reg', 'car_front', 'car_big_front', 'person', 'bicyclist',
           'motorcyclist', 'trafficlight', 'sign', 'licence', 'zebra_crossing']
seg_all = ['none']

det_dict = {
    'car': 'car',
    'bus': 'bus',
    'truck': 'truck',
    'car_reg': 'car_reg',
    'car_big_reg': 'car_big_reg',
    'car_front': 'car_front',
    'car_big_front': 'car_big_front',
    'person': 'person',
    'bicyclist': 'bicyclist',
    'motorcyclist': 'motorcyclist',
    'trafficlight': 'trafficlight',
    'sign': 'sign',
    'licence': 'licence',
    'arrow': '',
    'zebra_crossing': 'zebra_crossing',
    'bridge': '',
    'tricyclist': '',
    'sewer': '',
    'negative': '',
    'car_big': 'car',
    'preson': 'person'
}  # 20220923


def get_train_dataset_total():
    # to coco
    skip_num = 10
    CLASSES = ['face', 'body', 'hand']
    print(save_root)
    writer_train = JsonWriter(
        osp.join('/hzli/DSC_detect/Json', "dsc_train_new.json"), CLASSES,
        mode='w')
    train_ann_file = '/hzli/DSC_detect/ImageSets/Main/trainval.txt'
    img_prefix = '/hzli/DSC_detect/JPEGImages'
    ann_prefix = '/hzli/DSC_detect/Annotations'


    img_ids = mmcv.list_from_file(train_ann_file)
    from tqdm import tqdm
    total_names = []
    a=0
    for img_id in tqdm(img_ids):
        a+=1
        if a%3!=0:
            continue
        filename = '{}.jpg'.format(img_id)
        size = None
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_path = osp.join(img_prefix,
                                '{}.jpg'.format(img_id))
            img = Image.open(img_path)
            width, height = img.size

        xml_path = osp.join(ann_prefix, f'{img_id}.xml')

        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations_info=[]

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in total_names:
                total_names.append(name)
            print(total_names)
            if name not in CLASSES:
                continue
            # label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            # print(bbox)
            ignore = False

            annotations_info.append(
                addAnnItem(name, np.array(bbox).reshape(-1)))

        image_info = {}

        image_info["file_name"] = filename
        image_info["height"] = height
        image_info["width"] = width

        writer_train.insert_image(image_info, annotations_info)




    writer_train.write_json()



    writer_train = JsonWriter(
        osp.join('/hzli/DSC_detect/Json', "dsc_test_new.json"), CLASSES,
        mode='w')

    train_ann_file = '/hzli/DSC_detect/ImageSets/Main/test.txt'
    img_prefix = '/hzli/DSC_detect/JPEGImages'
    ann_prefix = '/hzli/DSC_detect/Annotations'


    img_ids = mmcv.list_from_file(train_ann_file)
    from tqdm import tqdm

    for img_id in tqdm(img_ids):
        filename = '{}.jpg'.format(img_id)
        size = None
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_path = osp.join(img_prefix,
                                '{}.jpg'.format(img_id))
            img = Image.open(img_path)
            width, height = img.size

        xml_path = osp.join(ann_prefix, f'{img_id}.xml')

        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations_info=[]

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CLASSES:
                continue
            # label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False

            annotations_info.append(
                addAnnItem(name, np.array(bbox).reshape(-1)))

        image_info = {}

        image_info["file_name"] = filename
        image_info["height"] = height
        image_info["width"] = width
        writer_train.insert_image(image_info, annotations_info)
    writer_train.write_json()






if __name__ == '__main__':
    dataroot = '/hzli/ADAS_DATASETS'
    ignore_label = False
    expand_width = 12
    if (ignore_label):
        det_keep = ['person', 'head', 'ignore']
    else:
        det_keep = det_all
    scene = [
        'ADAS_AllInOne_New',
        'ADAS',
    ]
    save_root = '/hzli/ADAS_DATASETS/sub_sets/adas_10'
    get_train_dataset_total()

    # save_root = '/hzli/ADAS_DATASETS/sub_sets/test'
    # get_test_dataset(scene)