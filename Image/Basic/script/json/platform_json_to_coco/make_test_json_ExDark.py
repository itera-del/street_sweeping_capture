import sys 
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from datasplit import *
from platform2coco import *

# 备忘
# 原数据标签
det_all=['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']
seg_all=['none']

det_dict = {
    'Bicycle': 'Bicycle', 
    'Boat': 'Boat', 
    'Bottle': 'Bottle', 
    'Bus': 'Bus',
    'Car': 'Car',
    'Cat': 'Cat',
    'Chair': 'Chair',
    'Cup': 'Cup',
    'Dog': 'Dog',
    'Motorbike': 'Motorbike',
    'People': 'People',
    'Table': 'Table',
}
                        
def get_dataset():

    writer_train = JsonWriter(osp.join(save_root, to_json_name), det_all, mode='w')
    images = get_files(os.path.join(dataroot, ori_jpg_name), target='.jpg')
    jsons = get_files(os.path.join(dataroot, ori_json_name), target='.json')

    out_pairs = get_img_label_pair(images, jsons)
    print('Img Len: {}, Json Len: {}, Pairs Lens: {}'.format(len(images),len(jsons), len(out_pairs)))

    target_parser = PlatformTargetParser(seg_all, det_all)
    for i, (img_name, json_name) in enumerate(out_pairs):

        if(json_name[-4:]!='json'):
            continue

        json_t = target_parser.parse(json_name)
        
        # mapping to coco
        image_info = {}
        img_name=img_name.split('/')[-1]
        image_info["file_name"] = '{}/{}'.format(ori_jpg_name, img_name)
        image_info["height"] = json_t["height"]
        image_info["width"] = json_t["width"]
        annotations_info = []
        for instacne in json_t["annotations"]:
            if instacne['type'] == 'rectangle':
                name=instacne["class_name"]
                if name not in det_dict:
                    print(instacne["class_name"],'|')
                    continue
                if(name!=''):
                    annotations_info.append(addAnnItem(det_dict[name], np.array(instacne["points"]).reshape(-1)))
        writer_train.insert_image(image_info, annotations_info)

    writer_train.write_json()


if __name__ == '__main__':
    dataroot = '/yuanhuan/data/image/Open_Source/ExDark/original/'
    save_root = '/yuanhuan/data/image/Open_Source/ExDark/original/'
    
    # # ori_jpg_name = "JPEGImages_train"
    # # ori_json_name = "Json"
    # # to_json_name = "train.json"
    
    # ori_jpg_name = "JPEGImages_val"
    # ori_json_name = "Json"
    # to_json_name = "val.json"

    # ori_jpg_name = "JPEGImages_train_RetinexFormer_LOL_v2_synthetic"
    # ori_json_name = "Json"
    # to_json_name = "train_RetinexFormer_LOL_v2_synthetic.json"
    
    ori_jpg_name = "JPEGImages_val_RetinexFormer_LOL_v2_synthetic"
    ori_json_name = "Json"
    to_json_name = "val_RetinexFormer_LOL_v2_synthetic.json"
    
    get_dataset()