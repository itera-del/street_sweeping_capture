import sys 
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from datasplit import *
from platform2coco import *

# 备忘
# 原数据标签
det_all=["char"]
seg_all=['none']
                        
def get_dataset():

    writer_train = JsonWriter(osp.join(save_root, to_json_name), det_all, mode='w')

    images = []
    jsons = []
    with open(os.path.join(dataroot, train_val_dataset), "r") as f:
        for line in f:
            images.append(os.path.join(dataroot, ori_jpg_name, line.strip() + '.jpg'))
            jsons.append(os.path.join(dataroot, ori_json_name, line.strip() + '.json'))

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
                if(name!=''):
                    annotations_info.append(addAnnItem(name, np.array(instacne["points"]).reshape(-1)))
        writer_train.insert_image(image_info, annotations_info)

    writer_train.write_json()


if __name__ == '__main__':
    dataroot = "/yuanhuan/data/image/RM_ANPR/original/RM_Character"
    save_root= "/yuanhuan/data/image/RM_ANPR/original/RM_Character"
    
    train_val_dataset = "ImageSets/Main/trainval.txt"
    ori_jpg_name = "JPEGImages"
    ori_json_name = "Jsons"
    to_json_name = "train.json"

    # train_val_dataset = "ImageSets/Main/test.txt"
    # ori_jpg_name = "JPEGImages"
    # ori_json_name = "Jsons"
    # to_json_name = "test.json"
    
    get_dataset()