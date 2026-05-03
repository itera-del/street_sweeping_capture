#!/root/anaconda3/bin bash
source /root/anaconda3/bin/activate base
conda init bash
conda activate py37_paddleocr
export PYTHONPATH=/yuanhuan/code/demo/Image/recognition2d/PaddleOCR/:$PYTHONPATH
cd /yuanhuan/code/demo/Image/recognition2d/PaddleOCR/
python3 -m paddle.distributed.launch --gpus '0,1'  tools/train.py -c configs/rec/rmai_ocr/china/chn_rm_cnn_tc_res.yml -o Global.pretrained_model=/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240625_w_aug_w_fangse_sample/iter_epoch_300.pdparams