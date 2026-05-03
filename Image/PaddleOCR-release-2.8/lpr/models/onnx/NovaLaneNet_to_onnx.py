import os
import sys
import torch

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
import models.NovaLaneNet as NovaLaneNet


if __name__ == '__main__':

    # 定义设备（优先使用 GPU，否则 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # LANENET_NOVT_TEST
    # # input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_20230209/"
    # input_dir = "/yuanhuan/model/image/lpr/morocco/seg_city_kind_num_morocco_20250515/"
    # pkl_model_name = "LaneNetNova_best.pth"
    # onnx_model_name = 'morocco_seg_city_kind_num_20250515.onnx'

    # # model
    # model = NovaLaneNet.LANENET_NOVT_TEST(num_classes=4)
    # state = torch.load(os.path.join(input_dir, pkl_model_name), map_location=device)
    # model_dict = model.state_dict()
    # model.load_state_dict(state['net'])
    # model.eval()

    # dummy_input = torch.randn(1, 3, 64, 128)
    # input_names = [ "data" ]
    # output_names = [ "prob" ]

    # torch.onnx.export(model, dummy_input, os.path.join(input_dir, onnx_model_name), input_names=input_names, output_names=output_names)

    # LANENET_NOVT_HEAD_SEG_CLA
    # input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230629/"
    # input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/"
    # input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230705/"
    # input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230714/"
    # input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_noaug_20250320/"
    # input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_aug_20250320/"
    # input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_noaug_20250408/"
    input_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_shate_aug_20250811/"
    # pkl_model_name = "LaneNetNova2Head.pth"
    pkl_model_name = "LaneNetNova2Head_best.pth"
    onnx_model_name = 'shate_seg_city_class_color_20250811.onnx'

    # model
    # model = NovaLaneNet.LANENET_NOVT_HEAD_SEG_CLA(num_classes_1_head=17, num_classes_2_head=4)
    # model = NovaLaneNet.LANENET_NOVT_HEAD_SEG_CLA(num_classes_1_head=18, num_classes_2_head=4)
    model = NovaLaneNet.LANENET_NOVT_HEAD_SEG_CLA(num_classes_1_head=18, num_classes_2_head=5)
    state = torch.load(os.path.join(input_dir, pkl_model_name), map_location=device)
    model_dict = model.state_dict()
    model.load_state_dict(state['net'])
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 128)
    input_names = [ "data" ]
    output_names = [ "prob", "prob_1" ]

    torch.onnx.export(model, dummy_input, os.path.join(input_dir, onnx_model_name), input_names=input_names, output_names=output_names)

    # # LANENET_NOVT_HEAD_SEG_CLA
    # input_dir = "/yuanhuan/model/image/lpr/cn/seg_color_cn_20230530/"
    # pkl_model_name = "LaneNetNova2Head.pth"
    # onnx_model_name = 'seg_color_cn_20230530.onnx'

    # # model
    # model = NovaLaneNet.LANENET_NOVT_HEAD_SEG_CLA(num_classes_1_head=1, num_classes_2_head=6)
    # state = torch.load(os.path.join(input_dir, pkl_model_name))
    # model_dict = model.state_dict()
    # model.load_state_dict(state['net'])
    # model.eval()

    # dummy_input = torch.randn(1, 3, 64, 128)
    # input_names = [ "data" ]
    # output_names = [ "prob", "prob_1" ]

    # torch.onnx.export(model, dummy_input, os.path.join(input_dir, onnx_model_name), input_names=input_names, output_names=output_names)