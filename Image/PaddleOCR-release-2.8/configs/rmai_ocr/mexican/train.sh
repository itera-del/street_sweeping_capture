#!/usr/bin/env bash
source /root/anaconda3/bin/activate base
conda activate py37_paddleocr

export PYTHONPATH=/ycliu/PaddleOCR/:$PYTHONPATH
cd /ycliu/PaddleOCR

CONFIG=configs/rec/rmai_ocr/mexican/mexican_rm_cnn_tc_res.yml
PRETRAIN=/ycliu/PaddleOCR/models/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202406_ALL_US/latest
RUN_TAG=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR=/ycliu/PaddleOCR/models//v1_mexican_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_${RUN_TAG}_1_1

# 单卡训练：直接使用 train.py，不走 paddle.distributed.launch
if [ -n "${PRETRAIN}" ]; then
  python3 tools/train.py -c "${CONFIG}" -o Global.pretrained_model="${PRETRAIN}" Global.save_model_dir="${SAVE_DIR}"
else
  python3 tools/train.py -c "${CONFIG}" -o Global.pretrained_model= Global.save_model_dir="${SAVE_DIR}"
fi