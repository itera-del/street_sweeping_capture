#!/usr/bin/env bash
# 对 debug 目录下裁剪图用 Paddle 与 infer.sh 相同的配置再推理一遍，
# 并将识别结果以尾缀 __paddle_{文本} 拼入文件名后重命名（与 tools/reinfer_rec_debug_rename.py 一致）。
#
# 使用前请按本机环境修改 conda 与路径。

set -e

source /root/anaconda3/bin/activate base
conda activate py37_paddleocr

export PYTHONPATH=/yuanhuan/code/demo/Image/recognition2d/PaddleOCR/:$PYTHONPATH
cd /yuanhuan/code/demo/Image/recognition2d/PaddleOCR/

CONFIG=configs/rec/rmai_ocr/mexican/mexican_rm_cnn_tc_res.yml
# 与 options_lpr_mexican.py 中 ONNX 同一次训练目录（20260415_092206），便于与 onnx 对齐对比
PRETRAINED=/yuanhuan/model/image/lpr/paddle_ocr/v1_mexican_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20260415_092206/latest
DEBUG_DIR=/yuanhuan/data/temp_result/test/test_20260420/debug

python3 tools/reinfer_rec_debug_rename.py \
  -c "${CONFIG}" \
  -o Global.pretrained_model="${PRETRAINED}" Global.infer_img="${DEBUG_DIR}"
