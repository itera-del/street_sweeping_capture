# dataset_huanwei.sh 完整解读（含 OCR 与 YoloX）

本文基于脚本 `dataset_huanwei.sh` 与当前项目中的真实样例数据，说明该脚本的数据流、训练准备流程，以及关键数据格式。

---

## 1. 脚本定位

`dataset_huanwei.sh` 是一个“数据生产流水线脚本”，主要将：

1. 闭环抽帧数据  
2. 平台标注结果  
3. 训练数据导出格式  

串成 OCR 与检测（YoloX）两条训练链路。

它的使用方式是：**按 step 取消/添加注释分段执行**，不是一次性全流程自动跑完。

---

## 2. 关键目录与变量

- `data_closed_dir`：闭环系统抽帧目录（输入源）
- `data_dir`：原始项目数据目录（按日期批次）
- `data_csv_dir`：标注检查后输出的 csv
- `data_crop_dir`：裁剪出的 ROI 图（常用于 OCR）
- `error_data_dir`：异常数据输出目录
- `analysis_dir`：统计分析输出目录
- `training_data_dir`：OCR 训练数据根目录
- `ocr_name`：OCR 数据版本名（如 `plate_huanwei_20251030`）

---

## 3. Step-by-step 流程

## Step 0：数据拷贝与平台格式准备（注释态）

目标：将闭环抽帧数据拷贝并转换到项目目录。

主要调用：

- `data_move.py`：拷贝 jpg + xml
- `xml_to_platform_json_HUANWEI.py`：将 xml 转平台 json

---

## Step 1：标注检查与统计（注释态）

目标：清洗并校验数据，生成结构化 csv 和错误样本。

主要调用：

- `dataset_csv.py`：输出 `*.csv`，并可写异常样本
- `analysis_dataset_label_num.py`：统计标签分布
- `analysis_dataset_label_num_merge.py`：多批次统计汇总

---

## Step 2：OCR 训练数据生成（注释态）

目标：从标注框裁出车牌图，生成 OCR 训练/验证拆分。

主要调用：

- `gen_ocr_img.py`：按 ROI 从大图裁车牌图
- `data_train_test_split_ocr.py`：OCR train/val 划分
- `gen_ocr_img_augment.py`：增强样本生成
- `data_train_test_split_ocr_augment.py`：增强样本划分
- `data_train_test_split_ocr_merge.py`：基础集与增强集合并

---

## Step 3：导出 PaddleOCR 标签（当前启用）

目标：把 OCR 数据导出成 PaddleOCR 训练可直接使用的形式。

主要调用：

- `lpr_to_paddleocr_label.py`：生成 `ImageSets/.../*.txt` 标签文件
- `label_dict.py`：生成字符字典（如 `huanwei_dict.txt`）

脚本当前配置为：

- `image_set_name=ImageSetsOcrLabelNoDiffsteNoAug`
- `paddle_ocr_data_dir=/yuanhuan/model/image/lpr/paddle_dict/plate_huanwei_20251030_NoDiffsteNoAug`

---

## Step 4：检测器（YoloX）训练准备（注释态）

目标：将平台标注转换为检测训练输入（VOC/manifest）。

主要调用：

- `platform_json_to_xml.py`：平台 json -> VOC xml
- `data_train_test_split.py`：检测 train/test 划分
- `run_manifest_from_txt_bash.sh`：生成训练 manifest
- `find_txt_data.py`：按 txt 拷测试图片

---

## 4. OCR 与 YoloX 的关系

这是两阶段任务：

1. **YoloX（检测）**：在整图中找到车牌框（bbox）
2. **OCR（识别）**：对框内车牌图识别字符

二者数据形态不同：

- 检测：`整图 + 类别 + 框坐标`
- OCR：`车牌小图 + 字符串标签`

---

## 5. 真实数据格式样例（来自当前项目）

## 5.1 CSV（OCR 输入源）

样例文件：

- `/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_csv/Argentina_20251021_有编号有垃圾桶.csv`

表头：

```text
img_path,json_path,id,name,roi,num
```

样例行：

```text
/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina/Argentina_20251021_有编号有垃圾桶/JPEGImages/1_4103706862338261011_0.jpg,/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina/Argentina_20251021_有编号有垃圾桶/Jsons/1_4103706862338261011_0.json,1,1_4103706862338261011_0-00_201417,"1223,1627,177,449",201417
```

字段说明：

- `img_path`：原图路径
- `json_path`：对应平台 json
- `id`：该图内目标序号
- `name`：实例名（含编号信息）
- `roi`：框坐标，格式为 `x1,x2,y1,y2`
- `num`：车牌字符标签（OCR GT）

---

## 5.2 平台 JSON（检测原始标注）

样例文件：

- `/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina/Argentina_20251021_有编号有垃圾桶/Jsons/1_4109688481008369683_30.json`

核心结构示例：

```json
{
  "img_name": "1_4109688481008369683_30.jpg",
  "width": 1920,
  "height": 1080,
  "shapes": [
    {
      "label": "license",
      "points": [1104, 182, 1542, 451],
      "attributes": [
        {"name": "status", "value": "n"},
        {"name": "number", "value": "201060"}
      ]
    },
    {
      "label": "garbage",
      "points": [1599, 0, 1919, 749]
    }
  ]
}
```

说明：

- `label=license` 是车牌目标
- `attributes.number` 常作为 OCR 标签来源
- `points` 是检测框坐标

---

## 5.3 VOC XML（YoloX 训练常见中间格式）

样例文件：

- `/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina/Argentina_20251021_有编号有垃圾桶/Annotations/2_4109725829776310291_40.xml`

核心结构示例：

```xml
<annotation>
  <filename>2_4109725829776310291_40.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
    <depth>3</depth>
  </size>
  <object>
    <name>garbage</name>
    <bndbox><xmin>290</xmin><ymin>358</ymin><xmax>721</xmax><ymax>834</ymax></bndbox>
  </object>
  <object>
    <name>license</name>
    <bndbox><xmin>390</xmin><ymin>600</ymin><xmax>444</xmax><ymax>688</ymax></bndbox>
  </object>
</annotation>
```

说明：

- 每个 `<object>` 是一个检测目标
- `name` 是类别，`bndbox` 是框坐标

---

## 5.4 PaddleOCR 训练标签 txt（最终 OCR 训练输入）

样例文件：

- `/yuanhuan/model/image/lpr/paddle_dict/plate_huanwei_20251030_NoDiffsteNoAug/ImageSets/Main/train.txt`

真实行格式：

```text
/yuanhuan/data/image/RM_HUANWEI/training_ocr/plate_huanwei_20251030/Argentina_20240912/Images/0000000000000000-240912-144352-165648-000006003920_150032_cut_154.0s_97.0s_fcnt_0150-00_473386.jpg	473386
```

说明：

- 左侧：OCR 小图路径
- 右侧：标签字符串
- 分隔符：`\t`（Tab）

---

## 5.5 OCR 字典文件

样例文件：

- `/yuanhuan/model/image/lpr/paddle_dict/plate_huanwei_20251030_NoDiffsteNoAug/huanwei_dict.txt`

样例内容：

```text
 
1
2
3
4
5
6
7
8
9
0
^
-
#
```

说明：

- 每行一个字符
- 训练与推理都需与该字典保持一致

---

## 6. 两条训练链路的数据流（简版）

## OCR 链路

`Json/CSV(含num)` -> `裁剪车牌图` -> `train/val txt(路径+字符串)` -> `字典` -> PaddleOCR 训练

## YoloX 链路

`平台Json` -> `VOC XML` -> `划分txt/manifest` -> YoloX 训练

---

## 7. 实操注意点

1. 先确认 `training_data_dir/$ocr_name` 已由 Step2 产出，否则 Step3 会找不到输入。
2. `roi` 在 csv 中是 `x1,x2,y1,y2`，和常见 `x1,y1,x2,y2` 顺序不同，处理脚本要一致。
3. OCR 目录后缀 `NoDiffsteNoAug` 需要和你实际版本一致，否则会混数据。
4. 检测与识别不要共用同一标注文件格式：YoloX 吃框，OCR 吃字符串。

---

## 8. 一句话总结

这个脚本核心是：**同一批平台标注，拆分成两种训练数据形态**——  
给 YoloX 的是“整图框标注”，给 OCR 的是“车牌小图+字符标签”。
