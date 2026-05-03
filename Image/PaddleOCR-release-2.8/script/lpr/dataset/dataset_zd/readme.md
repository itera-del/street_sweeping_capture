## 【数据准备】
1. 数据闭环和数据挖掘自动工具：
    - 使用 InterVL2 LLM 语言大模型挖掘特定车牌样式车牌
    - /yuanhuan/code/demo/Image/dataset_closed_loop/dataset_project/data_closing_capture.sh
2. 实习生人工复检：筛选有效图像素材，挑选需要标注的图片
3. dataset_zd.sh，step 0：数据拷贝：将需要标注的图片拷贝到对应的目录下
    - 数据拷贝：将需要标注的图片拷贝到对应的目录下
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_move/data_move.py
        - 某些通道的图像需要水平镜像
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_move/data_flip.py
    - 数据标注(xml 2 json)：
        - /yuanhuan/code/demo/Image/Basic/script/xml/xml_to_platform_json_LPR.py
        - 标注文件名称必须为 images，预标注标签为 json_v0，包含 meta.json 文件
4. 实习生标注：
    - 检修标签：['car', 'truck', 'bus', 'motorcycle', 'license']
    - 新增车牌数据标签：license 子属性，number\color\column\status
5. dataset_zd.sh，step 1：处理标注数据
    - 数据检查，生成 csv 文件，并裁剪 corp 车牌
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_csv.py
    - 分析数据分布
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_analysis/analysis_dataset_label_num.py
6. dataset_zd.sh，step 2：生成预标注标签 seg
    - seg 板端模型
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_pre_label/dataset_pre_label_seg_mask.py
    - seg 分割大模型
        - /yuanhuan/code/demo/Image/segmentation2d/mmsegmentation/demo/image_dir_demo_w_json.py
7. dataset_zd.sh，step 3：生成训练数据 seg
    - 数据检查，生成 csv 文件
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_crop_csv.py
    - 制作 seg mask
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_seg_mask.py
    - 制作 aug seg mask
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_seg_mask_augment.py
8. dataset_zd.sh，step 5：生成训练数据 ocr
    - 生成 OCR img：
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_ocr_img.py
    - 生成 OCR aug img：
        - /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_ocr_img_augment.py
9. dataset_zd.sh，step 5：生成预标注标签 ocr（样本生成 char mask）
    - 参考 README 文件：Image/sd/script/diffste/dataset_plate/readme.md
    - resize & padding，处理成检测大模型的输入
        - /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_pretreatment/dataset_resize_padding.py
    - 检测大模型：
    - 转换为平台标注格式：
        - /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_xml/xml_to_platform_json_LPR_zd_char_bbox.py