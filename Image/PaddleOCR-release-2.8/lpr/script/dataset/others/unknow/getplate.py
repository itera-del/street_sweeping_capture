import os
import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET

class Transform(object):
    def __init__(self):
        pass
    def __call__(self, target):
        box = np.empty((0, 4))
        plate = []
        for obj in target.iter('object'):
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) # - 1
                bndbox.append(cur_pt)
            try:
                p = obj.find('vehicleic').text
            except:
                p = None
            box = np.vstack((box, bndbox))
            plate.append(p)
        return box, plate

def save_plates(img_path, xml_path, num):
    img = cv2.imread(img_path)
    xml = ET.parse(xml_path).getroot()
    boxes, plates = TRANS(xml)
    for i in range(len(boxes)):
        if plates[i] is None:
            continue
        else:
            box = np.array(boxes[i], dtype=int)
            xmin, ymin, xmax, ymax = box
            p = img[ymin:ymax, xmin:xmax, :]
            cv2.imwrite('{}/{}__{}.jpg'.format( PATH, str(num),plates[i]), p)
    return 0

if __name__ == '__main__':
    TRANS = Transform()
    PATH = "/home/workspace/xnlin/xnlin_data/Mexican/train/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    file = glob.glob("/opt/8091/external/plate/license_plate/samplepics/Mexican/20210628/*.xml")
    num = 0
    for i, xml_name in enumerate(file):
        img_name = xml_name[:-4]+".jpg"
        save_plates(img_name, xml_name,num)
        num = num+1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


