import argparse
from engine import LPR, CitySeg, ColorClass, CityColorSeg, CityMultiLabelSeg, CitySegColorClass, BeltSeg, BsdSeg, BsdRFSeg


def parse_args():

    parser = argparse.ArgumentParser()
    # phase
    # parser.add_argument('--phase'     , type=str  , default='cityseg' , help='CitySeg')                                   # 城市分割
    # parser.add_argument('--phase'     , type=str  , default='colorseg' , help='CitySeg')                                  # 颜色分割
    # parser.add_argument('--phase'     , type=str  , default='colorclass' , help='ColorClass')                             # 颜色分类
    # parser.add_argument('--phase'     , type=str  , default='citycolorseg' , help='CityColorSeg')                         # 城市分割、颜色分割，2Head Softamx（abolish）
    # parser.add_argument('--phase'     , type=str  , default='citymultilabelseg' , help='CityMultiLabelSeg')               # 城市分割、颜色分割，Sigmoid（abolish）
    parser.add_argument('--phase'     , type=str  , default='citysegcolorclass' , help='CitySegColorClass')               # 城市分割、颜色分类
    # parser.add_argument('--phase'     , type=str  , default='citycolorclassseg_no_grad' , help='CityColorcClassSeg')      # 城市分割、颜色分类，颜色梯度冻结（abolish）
    # parser.add_argument('--phase'     , type=str  , default='ocr'     , help='ocr')                                       # OCR 识别
    # Hyperparameters
    # parser.add_argument('--batchsize' , type=int  , default=512       , help='train batchsize')
    parser.add_argument('--batchsize' , type=int  , default=256       , help='train batchsize')

    # debug
    # parser.add_argument('--batchsize' , type=int  , default=1       , help='train batchsize')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.phase=='ocr':
        #######
        # zd
        #######
        # ocr_list_path = "/yuanhuan/data/image/RM_ANPR/training/plate_zd_mask_202301/"
        # ocr_list_path = "/yuanhuan/data/image/RM_ANPR/training/plate_zd_mask_202307/ImageSetsOcrLabelNoAug"
        ocr_list_path = "/yuanhuan/data/image/RM_ANPR/training/plate_zd_mask_202307/ImageSetsOcrLabel"
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_all_UAE_hisi_1010/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_1111/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_1120/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230119/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230209/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_tcres_20230209/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_tcres_20230217/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230217/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230301/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, padding_bool=True, data_dict_name="script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_normal", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230630/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, padding_bool=True, early_stop=100, data_dict_name="script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_normal", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230703/")
        trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, padding_bool=True, early_stop=100, data_dict_name="script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_normal", savepath="/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230705/")
        trainEngine()
        
        # #######
        # # brazil
        # #######
        # ocr_list_path = "/yuanhuan/data/image/RM_ANPR/training/plate_brazil/"
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_20230111/")
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_first_line_20230116/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_first_line_20230117/")
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_second_line_20230116/")
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_second_line_20230117/")
        # # ocr_list_path = "/yuanhuan/data/image/RM_ANPR/training/plate_brazil_2_line/"
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_2_line_20230111/")
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_2_line_20230112/")
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_2_line_20230115/")
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, data_dict_name="lpr.script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict", savepath="/yuanhuan/model/image/lpr/brazil/ocr_brazil_2_line_256_32_20230116/")
        # trainEngine()

        # #######
        # # cn
        # #######
        # ocr_list_path = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/ImageSetsOcrLabelNoAug/"
        # # ocr_list_path = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/ImageSetsOcrLabel/"
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, padding_bool=False, early_stop=300, data_dict_name="script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_normal", savepath="/yuanhuan/model/image/lpr/cn/ocr_cn_20230530/")
        # # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, padding_bool=False, early_stop=100, data_dict_name="script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_normal", savepath="/yuanhuan/model/image/lpr/cn/ocr_cn_20230602/")
        # trainEngine = LPR.TrainingEngine(ocr_list_path, args.batchsize, gray_bool=False, padding_bool=False, early_stop=300, data_dict_name="script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_normal", savepath="/yuanhuan/model/image/lpr/cn/ocr_bgr_cn_20230607/")
        # trainEngine()
    
    elif args.phase == 'cityseg':
        # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd/city_merge_train/ImageSets/Main/"
        # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd/city_merge_train_1210/ImageSets/Main/"
        # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202301/ImageSetsCityLabel/ImageSets/Main/"
        # cityseg = CitySeg.CitySegTrainingEngine(cityseglistpath, args.batchsize, seg_dict_name="script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", save_path="/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1019/")
        # cityseg = CitySeg.CitySegTrainingEngine(cityseglistpath, args.batchsize, seg_dict_name="script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", save_path="/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1110/")
        # cityseg = CitySeg.CitySegTrainingEngine(cityseglistpath, args.batchsize, seg_dict_name="script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", save_path="/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1220/")
        # cityseg = CitySeg.CitySegTrainingEngine(cityseglistpath, args.batchsize, seg_dict_name="script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_city", save_path="/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_20230209/")
        # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_Morocco_202504/ImageSetsLabelNoAug/city_label/ImageSets/Main"
        cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_Morocco_202504/ImageSetsLabel/city_label/ImageSets/Main"
        cityseg = CitySeg.CitySegTrainingEngine(cityseglistpath, args.batchsize, seg_dict_name="script.lpr.dataset.dataset_morocco.dataset_dict.dataset_morocco_dict_city", save_path="/yuanhuan/model/image/lpr/morocco/seg_city_kind_num_morocco_noaug_20250515/")
        cityseg()
    elif args.phase == 'colorseg':
        # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd/color_merge_train/ImageSets/Main/"
        cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd/color_merge_train_1210/ImageSets/Main/"
        # cityseg = CitySeg.CitySegTrainingEngine(cityseglistpath, args.batchsize, seg_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_color_zd_1101/")
        # cityseg = CitySeg.CitySegTrainingEngine(cityseglistpath, args.batchsize, seg_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_color_zd_1104/")
        cityseg = CitySeg.CitySegTrainingEngine(cityseglistpath, args.batchsize, seg_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_color_zd_1220/")
        cityseg()
    elif args.phase == 'colorclass':
        # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_cn_202305/ImageSetsColorLabelNoAug/ImageSets/Main/"
        cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_cn_202305/ImageSetsColorLabel/ImageSets/Main/"
        # cityseg = ColorClass.ColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_color_dict_name="dataset_cn_dict_color", save_path="/yuanhuan/model/image/lpr/cn/seg_color_cn_20230516/")
        cityseg = ColorClass.ColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_color_dict_name="dataset_cn_dict_color", save_path="/yuanhuan/model/image/lpr/cn/seg_color_cn_20230530/")
        cityseg()
    elif args.phase == 'citycolorseg':
        cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd/city_color_merge_train/ImageSets/Main/"
        # cityseg = CityColorSeg.CityColorSegTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_zd_1103/")
        cityseg = CityColorSeg.CityColorSegTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_zd_1104/")
        cityseg()
    elif args.phase == 'citymultilabelseg':
        cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_mutil_seg/sigmoid_merge_train/ImageSets/Main/"
        cityseg = CityMultiLabelSeg.CityMultiLabelSegTrainingEngine(cityseglistpath, args.batchsize, save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_mutil_seg_zd_1107/")
        cityseg()
    elif args.phase == 'citysegcolorclass':
        # # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202301/ImageSetsCityColorLabel/ImageSets/Main/"
        # # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202307/ImageSetsLabelNoAug/city_color_label/ImageSets/Main/"
        # # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202307/ImageSetsLabel/city_color_label/ImageSets/Main/"
        # # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202502/ImageSetsLabelNoAug/city_color_label/ImageSets/Main/"
        # # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202502/ImageSetsLabel/city_color_label/ImageSets/Main/"
        # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202504/ImageSetsLabelNoAug/city_color_label/ImageSets/Main/"
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1107/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1117/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1210/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1216/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230119/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230217/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230629/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230705/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_noaug_20250320/")
        # # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_aug_20250320/")
        # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city_2022", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_noaug_20250408/")
        cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_shate_202508/ImageSetsLabel/city_color_label/ImageSets/Main/"
        cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_shate_aug_20250811/")
        cityseg()

        # lmdb（速度慢，原因未知）
        # cityseglistpath = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202307/ImageSetsLabel/city_color_label/ImageSets/Main/"
        # lmdb_path = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_lmdb_202307/"
        # cityseglistpath = "/root/ImageSetsLabel/city_color_label/ImageSets/Main/"
        # lmdb_path = "/root/seg_zd_lmdb_202307/"
        # cityseg = CitySegColorClass.CitySegColorClasTrainingEngine(cityseglistpath, args.batchsize, seg_city_dict_name="dataset_zd_dict_city", seg_color_dict_name="dataset_zd_dict_color", save_path="/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230628/", lmdb_bool=True, lmdb_path=lmdb_path)
        # cityseg()
    elif args.phase == 'beltseg':
        beltsegtrainlist = "./input/dataset_belt/"
        beltseg = BeltSeg.BeltSegTrainingEngine(beltsegtrainlist,args.batchsize)
        beltseg()
    elif args.phase == 'handseg':
        beltsegtrainlist = "./input/dataset_hand/"
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
    elif args.phase == 'bsdrfseg':
        beltsegtrainlist = "./input/dataset_bsdrfseg/"
        beltseg = BsdRFSeg.BsdSegTrainingEngine(beltsegtrainlist,args.batchsize)
        beltseg()
    elif args.phase == 'islandseg':
        beltsegtrainlist = "./input/dataset_island/"
        beltseg = BsdSeg.BsdSegTrainingEngine(beltsegtrainlist,args.batchsize)
        beltseg()
    else:
        print("args.phase error!")
