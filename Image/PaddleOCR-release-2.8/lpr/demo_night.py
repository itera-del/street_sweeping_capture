
import os
import argparse
from engine import LPR, CitySeg, BeltSeg, BsdSeg
import engine
def parse_args():
    desc = 'License Recognition'
    parser = argparse.ArgumentParser()
    # phase
    parser.add_argument('--phase'     , type=str  , default='ocr'   , help='ocr/cityseg')
    # Hyperparameters
    parser.add_argument('--inputshape', type=list , default=[3,128,64], help='input shape of plate')
    parser.add_argument('--nclass'    , type=int  , default=1         , help='tot num of classes')
    parser.add_argument('--batchsize' , type=int  , default=128       , help='train batchsize')
    return parser.parse_args()
imglist=[]
imglisttest=[]
# # samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/RAK/Double"
# # imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/RAK/Single"
# # imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/AJMAN/Double"
# # imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/AJMAN/Single"
# # imglist.append(samplePath)

# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/others2/Single"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/others2/Double"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/new/Single"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/new/Double"
# imglist.append(samplePath)

# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/gen/imgs_double"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/gen/imgs_single"
# imglist.append(samplePath)

# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/SHJ/Double"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city/SHJ/Single"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Single/red/"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Single/yellow/"
# imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Single/white/"
# # imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Single/green/"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Single/others/"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Double/red/"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Double/yellow/"
# imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Double/white/"
# # imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Double/green/"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate1/Double/others/"
# imglist.append(samplePath)

# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Single/red/"
# # imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Single/yellow/"
# imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Single/white/"
# # imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Single/green/"
# imglisttest.append(samplePath)
# # # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Single/others/"
# # # imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Double/red/"
# # imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Double/yellow/"
# imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Double/white/"
# # imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Double/green/"
# imglisttest.append(samplePath)
# # # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Double/others/"
# # # imglisttest.append(samplePath)


# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate2/Single/red/"
# # imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate2/Single/yellow/"
# # imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Single/white/"
# # imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate2/Single/green/"
# # imglisttest.append(samplePath)
# # # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Single/others/"
# # # imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate2/Double/red/"
# # imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate2/Double/yellow/"
# # imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Double/white/"
# # imglisttest.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate2/Double/green/"
# # imglisttest.append(samplePath)
# # # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/plate3/Double/others/"
# # # imglisttest.append(samplePath)

# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/night3/"
# # imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/night6/"
# # imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/night7/"
# # imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/gen/double/"
# imglist.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/gen/single/"
# imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/night/"
# # imglist.append(samplePath)
# # samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/night2/"
# # imglisttest.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/clear/"
imglist.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/night11/"
imglist.append(samplePath)
samplePath ="/home/workspace/xnlin/xnlin_data/DUBAI/night11/"
imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city2/SHJ/Double"
# imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city2/SHJ/Single"
# imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city2/new/Single"
# imglisttest.append(samplePath)
# samplePath ="/home/workspace/xnlin/xnlin_data/plate_zd_city2/new/Double"
# imglisttest.append(samplePath)
#########################################################################




if __name__ == '__main__':
    args = parse_args()
    if args.phase=='ocr':
        trainEngine = LPR.TrainingEngine(imglist, imglisttest, args.batchsize, "./output/dubai/all_time_lstm/")
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
