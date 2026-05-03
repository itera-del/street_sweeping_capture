#!/usr/bin/env bash
source /root/anaconda3/bin/activate base
conda activate py37_paddleocr

set -e

export PYTHONPATH=/yuanhuan/code/demo/Image/recognition2d/PaddleOCR/:$PYTHONPATH
cd /yuanhuan/code/demo/Image/recognition2d/PaddleOCR/

CONFIG=configs/rec/rmai_ocr/mexican/mexican_rm_cnn_tc_res.yml
PRETRAINED=/yuanhuan/model/image/lpr/paddle_ocr/v1_mexican_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20260414_114526/latest
SAVE_RES_PATH=/ycliu/PaddleOCR/data/mexican_lpr/Mexican_20260331_frames_5f_10060/output_ocr/predict_mexican_testset.txt

# 按 split 生成软链接目录后再推理（可选: train/test）。
SPLIT=test
LIST_FILE_BASE=/ycliu/PaddleOCR/data/mexican_lpr/Mexican_20260331_frames_5f_10060/paddle_dict_0414/ImageSets/Main
LIST_FILE="${LIST_FILE_BASE}/${SPLIT}.txt"
LINK_ROOT=/tmp/paddleocr_infer_links
LINK_DIR="${LINK_ROOT}/mexican_${SPLIT}"

if [[ ! -f "${LIST_FILE}" ]]; then
  echo "[ERROR] list file not found: ${LIST_FILE}"
  exit 1
fi

rm -rf "${LINK_DIR}"
mkdir -p "${LINK_DIR}"

# 从 train/test 列表逐行读取绝对路径并创建软链接。
# 列表格式: <img_abs_path>\t<label>
while IFS=$'\t' read -r img_path _ || [[ -n "${img_path}" ]]; do
  [[ -z "${img_path}" ]] && continue
  if [[ -f "${img_path}" ]]; then
    ln -sf "${img_path}" "${LINK_DIR}/$(basename "${img_path}")"
  fi
done < "${LIST_FILE}"

mkdir -p "$(dirname "${SAVE_RES_PATH}")"

python3 tools/infer_rec.py \
  -c "${CONFIG}" \
  -o Global.infer_img="${LINK_DIR}" Global.pretrained_model="${PRETRAINED}" Global.save_res_path="${SAVE_RES_PATH}"
