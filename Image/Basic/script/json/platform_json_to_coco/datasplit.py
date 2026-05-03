import os
import cv2
import sys
from random import shuffle, seed
from shutil import copy, move
from tqdm import tqdm

sys.path.insert(0, "/yuanhuan/code/demo/Image/")
from Basic.script.json.platform_json_to_coco.precheck import get_data_id_list


def writelist(datalist, listpath):
    with open(listpath, "w+") as f:
        for i in datalist:
            f.write(i)
            f.write("\n")
    return 0


def readlist(filepath):
    datalist = []
    with open(filepath, "r+") as f:
        for i in f:
            datalist.append(i.strip())
    return datalist


def make_block(datalist, len_block=10):
    newlist = []
    tmp = []
    if len(datalist) % len_block == 0:
        count_block = len(datalist) // len_block
    else:
        count_block = len(datalist) // len_block + 1

    for i in range(count_block):
        tmp += datalist[i * len_block : (i + 1) * len_block]
        newlist.append(tmp)
        tmp = []
    return newlist


def blocks2list(datalist_block):
    newlist = []
    for i in range(len(datalist_block)):
        newlist += datalist_block[i]
    return newlist


def train_test_split(datalist, sizeblock, sprate=0.9):
    datalist_block = make_block(datalist, sizeblock)
    seed(10)
    shuffle(datalist_block)
    newdatalist = blocks2list(datalist_block)
    n = len(newdatalist)
    trainlist = newdatalist[: int(n * sprate)]
    testlist = newdatalist[int(n * sprate) :]
    return newdatalist, trainlist, testlist


def buildDir(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)


from time import sleep


def move_data(datadir, dataislist, new_imgdir, new_tagdir, check=False):
    # if '_' in datadir:
    #     datadir = datadir.replace('/', '_')
    for i in tqdm(range(len(dataislist))):

        # if i%100==0:
        #     sleep(10)

        src_path_img = "{}/images/{}.jpg".format(datadir, dataislist[i])
        new_path_img = "{}/{}.jpg".format(new_imgdir, dataislist[i])
        src_path_ann = "{}/json_v0/{}.json".format(datadir, dataislist[i])
        new_path_ann = "{}/{}.json".format(new_tagdir, dataislist[i])

        if check:
            img = cv2.imread(src_path_img)
            if img is None:
                pass
            else:
                copy(src_path_img, new_path_img)
                copy(src_path_ann, new_path_ann)
        else:
            copy(src_path_img, new_path_img)
            copy(src_path_ann, new_path_ann)

    return


if __name__ == "__main__":

    date = [
        "NewDaily/20220425",
        "NewDaily/20220429",
        "NewDaily/20220512",
        "ADASDetSeg",
        "BSDsidewaysSegOld",
    ]

    for d in date:
        datadir = "/lirui/DATA/BSDsideways/{}".format(d)
        dataid_list = get_data_id_list(datadir)
        _, trainid, testid = train_test_split(dataid_list, 50, 0.85)

        train_data_img_dir = "{}/train/".format(datadir)
        train_data_tag_dir = "{}/train/".format(datadir)
        test_data_img_dir = "{}/test/".format(datadir)
        test_data_tag_dir = "{}/test/".format(datadir)

        buildDir(train_data_img_dir)
        buildDir(train_data_tag_dir)
        buildDir(test_data_img_dir)
        buildDir(test_data_tag_dir)

        move_data(datadir, trainid, train_data_img_dir, train_data_tag_dir)
        move_data(datadir, testid, test_data_img_dir, test_data_tag_dir)
