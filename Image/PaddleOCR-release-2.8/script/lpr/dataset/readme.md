# 注：重庆集群，node2\node3无法训练，可能环境有问题

# 环境配置
# conda activate py37_paddleocr
# cd /yuanhuan/code/demo/Image/recognition2d/PaddleOCR/

# 模型训练
# python3 tools/train.py -c configs/rec/rmai_ocr/eu/eu_rm_cnn_tc_res.yml -o Global.pretrained_model=
# python3 -m paddle.distributed.launch --gpus '0,1'  tools/train.py -c configs/rec/rmai_ocr/eu/eu_rm_cnn_tc_res.yml -o Global.pretrained_model=