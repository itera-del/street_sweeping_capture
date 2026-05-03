#!/usr/bin/env bash
source /root/anaconda3/bin/activate base
conda activate py37_paddleocr

set -e

export PYTHONPATH=/yuanhuan/code/demo/Image/recognition2d/PaddleOCR/:$PYTHONPATH
cd /yuanhuan/code/demo/Image/recognition2d/PaddleOCR/

CONFIG=configs/rec/rmai_ocr/mexican/mexican_rm_cnn_tc_res.yml
# CHECKPOINT=/yuanhuan/model/image/lpr/paddle_ocr/v1_mexican_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20260414_114526/latest
# CHECKPOINT=/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202406_ALL_US/iter_epoch_500
CHECKPOINT=/yuanhuan/model/image/lpr/paddle_ocr/v1_mexican_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20260428_131110/latest

python3 tools/eval.py \
  -c "${CONFIG}" \
  -o Global.checkpoints="${CHECKPOINT}"