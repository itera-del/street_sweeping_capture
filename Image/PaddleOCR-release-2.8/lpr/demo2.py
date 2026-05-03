
import os
import argparse
from engine import LPR2, CitySeg, BeltSeg, BsdSeg
import engine
def parse_args():
    desc = 'License Recognition'
    parser = argparse.ArgumentParser()
    # phase
    parser.add_argument('--phase'     , type=str  , default='ocr'   , help='ocr/cityseg')
    # Hyperparameters
    parser.add_argument('--inputshape', type=list , default=[3,128,64], help='input shape of plate')
    parser.add_argument('--nclass'    , type=int  , default=1         , help='tot num of classes')
    parser.add_argument('--batchsize' , type=int  , default=128      , help='train batchsize')
    return parser.parse_args()
imglist=[]
imglisttest=[]
samplePath ="/home/workspace/xnlin/xnlin_data/Turkey2/dst3/"
imglist.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/Turkey2/dst2/"
imglist.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/Turkey2/dst5/"
imglist.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/Turkey2/dst6/"
imglist.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/Turkey2/test/"
imglist.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/Turkey/"
imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/Turkey2/dst4/"
# imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/Turkey2/test2/"
# imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/Turkey2/dst/"
# imglisttest.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/Mexican/train/"
imglist.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/Mexican/test/"
imglisttest.append(samplePath)

#########################################################################



 
if __name__ == '__main__':
    args = parse_args()
    if args.phase=='ocr':
        trainEngine = LPR2.TrainingEngine(imglist, imglisttest, args.batchsize, "./output/mexico/")
        trainEngine()
    elif args.phase == 'cityseg':
        citysegtrainlist = "./input/dataset_city/"
        cityseg = CitySeg.CitySegTrainingEngine(citysegtrainlist,args.batchsize)
        cityseg()
    elif args.phase == 'beltseg':
        beltsegtrainlist = "./input/dataset_belt/"
        beltseg = BeltSeg.BeltSegTrainingEngine(beltsegtrainlist,args.batchsize)
        beltseg()
    elif args.phase == 'beltcls':
        beltsegtrainlist = "./input/dataset_beltcls/"
        beltseg = BeltSeg.BeltClsEngine(beltsegtrainlist,args.batchsize)
        beltseg()
    elif args.phase == 'bsdseg':
        beltsegtrainlist = "./input/dataset_bsdseg/"
        beltseg = BsdSeg.BsdSegTrainingEngine(beltsegtrainlist,args.batchsize)
        beltseg()
    else:
        print("args.phase error!")
