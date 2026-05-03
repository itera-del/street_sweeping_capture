import json
import os.path as osp


class JsonWriter(object):
    def __init__(self, filename, class_names, description="Taxi 2021 all in one Dataset", version="1.0", mode='w'):
        assert mode == 'w' or mode == 'w+'
        self.filename = filename
        self.info = {"description": description, "version": version}

        self.class_dicts = {}
        self.categories = []
        for i, cls in enumerate(class_names):
            self.class_dicts[cls] = i
            self.categories.append({"id": i, "name": cls})

        if osp.exists(self.filename) and mode == 'w+':
            with open(self.filename, 'r') as f:
                content = json.load(f)
            self.images = content["images"]
            self.annotations = content["annotations"]
            if(len(self.images)):
                self.img_id = self.images[-1]["id"]
                self.ann_id = self.annotations[-1]["id"]
            else:
                self.img_id=0
                self.ann_id=0
        else:
            self.images = []
            self.annotations = []

            self.img_id = 0
            self.ann_id = 0

    def insert_image(self, image, annotations):
        # image 
        self.img_id += 1

        image_info = {"file_name": image["file_name"], "height": image["height"], "width": image["width"], "id": self.img_id}
        self.images.append(image_info)

        for anno in annotations:
            self.ann_id += 1
            area = anno["area"]
            bbox = anno["bbox"]
            if anno["class_name"] in self.class_dicts:
                category_id = self.class_dicts[anno["class_name"]]
                if "segmentation" in anno:
                    segmentation = anno["segmentation"]
                    anno_info = {"segmentation": segmentation, "area": area, "iscrowd": 0, "image_id": self.img_id, 
                                "bbox": bbox, "category_id": category_id, "id": self.ann_id}
                else:
                    anno_info = {"area": area, "iscrowd": 0, "image_id": self.img_id, 
                                "bbox": bbox, "category_id": category_id, "id": self.ann_id}   
                self.annotations.append(anno_info)
            # elif(anno["class_name"]=='ignore' and "segmentation" not in anno):
            #     category_id = -1
            #     anno_info = {"area": area, "iscrowd": 0, "image_id": self.img_id, 
            #                     "bbox": bbox, "category_id": category_id, "id": self.ann_id}   
            #     self.annotations.append(anno_info)
    def write_json(self):
        content = dict()
        content["info"] = self.info
        content["images"] = self.images
        content["annotations"] = self.annotations
        content["categories"] = self.categories

        with open(self.filename, "w") as outfile:
            json.dump(content, outfile)
