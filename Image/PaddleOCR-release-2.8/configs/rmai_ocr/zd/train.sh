#!/root/anaconda3/bin bash
source /root/anaconda3/bin/activate base
conda init bash
conda activate py37_paddleocr

export PYTHONPATH=/yuanhuan/code/demo/Image/recognition2d/PaddleOCR/:$PYTHONPATH
cd /yuanhuan/code/demo/Image/recognition2d/PaddleOCR/

# python3 -m paddle.distributed.launch --gpus '0,1'  tools/train.py -c configs/rec/rmai_ocr/zd/zd_rm_cnn_tc_res.yml -o Global.pretrained_model=/yuanhuan/model/image/lpr/paddle_ocr/v1_zd_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20250327_Aug/latest
# python3 -m paddle.distributed.launch --gpus '0,1'  tools/train.py -c configs/rec/rmai_ocr/zd/zd_rm_cnn_tc_res_aug.yml -o Global.pretrained_model=/yuanhuan/model/image/lpr/paddle_ocr/v1_zd_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20250408_Aug_Diffste/latest
# python3 -m paddle.distributed.launch --gpus '0,1'  tools/train.py -c configs/rec/rmai_ocr/zd/zd_rm_cnn_tc_res_aug.yml -o Global.pretrained_model=/yuanhuan/model/image/lpr/paddle_ocr/v1_zd_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20250408_Aug_Diffste/latest
python3 -m paddle.distributed.launch --gpus '0,1'  tools/train.py -c configs/rec/rmai_ocr/zd/zd_rm_cnn_tc_res_aug.yml -o Global.pretrained_model=/yuanhuan/model/image/lpr/paddle_ocr/v1_zd_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20250811_Aug/latest