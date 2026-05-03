import os
import cv2
import glob
import random
import numpy as np
import xml.etree.ElementTree as ET

class Transform(object):
    def __init__(self):
        pass
    def __call__(self, target):
        box_hand = np.empty((0, 4))
        box_wheel = np.empty((0, 4))
        for obj in target.iter('object'):
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) # - 1
                bndbox.append(cur_pt)
            try:
                p = obj.find('name').text
            except:
                continue
            if p =='hand':
                box_hand = np.vstack((box_hand, bndbox))
            elif p =='wheel':
                box_wheel = np.vstack((box_wheel, bndbox))
            elif p =='body':
                pass
            elif p =='phone':
                pass
            else:
                print(p)
        return box_hand, box_wheel

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


def save_hands(img_path, xml_path, path, num):
    img = cv2.imread(img_path)
    #print (img_path, img.shape)
    xml = ET.parse(xml_path).getroot()
    boxes_hand, boxes_wheel = TRANS(xml)
    for i in range(len(boxes_wheel)):
        wheelbox = np.array(boxes_wheel[i], dtype=int)
        xmin, ymin, xmax, ymax = wheelbox
        xmin = max(0,xmin+random.randint(-100, 20))
        ymin = max(0,ymin+random.randint(-100, 20))
        xmax = min(img.shape[1],xmax+random.randint(-20, 100))
        ymax = min(img.shape[0],ymax+random.randint(-20, 100))
        img_wheel = img[ymin:ymax, xmin:xmax, :]
        img_wheel_mask = np.zeros(img_wheel.shape, dtype=img_wheel.dtype)
        for i in range(len(boxes_hand)):
            handbox = np.array(boxes_hand[i], dtype=int)
            handxmin, handymin, handxmax, handymax = handbox
            if handxmax>=xmin and handxmin<=xmax and handymax>=ymin and handymin<=ymax:
                handxmin = max(handxmin, xmin) - xmin
                handymin = max(handymin, ymin) - ymin
                handxmax = min(handxmax, xmax) - xmin
                handymax = min(handymax, ymax) - ymin
                w = handxmax - handxmin
                h = handymax - handymin
                handxmin = int(handxmin + 0.2 * w)
                handxmax = int(handxmax - 0.2 * w)
                handymin = int(handymin + 0.2 * h)
                handymax = int(handymax - 0.2 * h)
                #cv2.rectangle(img_wheel, (handxmin,handymin), (handxmax,handymax), (255,255,255), 3)
                img_wheel_mask[handymin:handymax, handxmin:handxmax,:] = [1,1,1]
        cv2.imwrite('{}/src/{}_src.jpg'.format(path, str(num)), img_wheel)
        cv2.imwrite('{}/mask/{}_mask.png'.format(path, str(num)), img_wheel_mask)


if __name__ == '__main__':
    TRANS = Transform()
    PATH = "/home/workspace/xnlin/xnlin_data/SC/train/"
    IMGPATH = "/opt/8091sampleclips2/inside/active_safety/lirui/SmartCockpit/VOC_selftest/JPEGImages/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    file = glob.glob("/opt/8091sampleclips2/inside/active_safety/lirui/SmartCockpit/VOC_selftest/Annotations/*.xml")
    num = 0
    for i, xml_name in enumerate(file):
        names=xml_name.strip().split("/")
        img_name = names[-1][:-4]+".jpg"
        #print (IMGPATH + img_name)
        save_hands(IMGPATH + img_name, xml_name, PATH ,num)
        num = num+1
        # if num>100:
        #     break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


