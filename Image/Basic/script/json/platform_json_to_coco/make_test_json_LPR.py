import sys 
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from datasplit import *
from platform2coco import *

# 备忘
# 原数据标签
# det_all=['car', 'bus', 'truck', 'bicyclist', 'motorcyclist', 'license']
det_all=['car', 'bus', 'truck', 'motorcyclist', 'license']
# det_all=['car', 'bus', 'truck']
seg_all=['none']

det_dict = {
    'car': 'car', 
    'bus': 'bus', 
    'truck': 'truck', 
    'bicyclist': 'bicyclist',
    'motorcyclist': 'motorcyclist',
    'license': 'license',
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
    dataroot = '/yuanhuan/data/image/RM_Capture/original_c27_dupes_0_95/Balanced_selection/1k/'
    save_root= '/yuanhuan/data/image/RM_Capture/original_c27_dupes_0_95/Balanced_selection/1k/'
    
    ori_jpg_name = "JPEGImages"
    ori_json_name = "Jsons"
    to_json_name = "train.json"
    
    # ori_jpg_name = "JPEGImages_test"
    # ori_json_name = "Json"
    # to_json_name = "test.json"

    # ori_jpg_name = "JPEGImages_dupes_0_95_test"
    # ori_json_name = "Json_dupes_0_95_test_all"
    # to_json_name = "test.json"
    
    get_dataset()