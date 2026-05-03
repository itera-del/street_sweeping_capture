import numpy as np 
import json

face_dict = {"rc_attr":{"confidence":[0.0,0.051025390625,0.91015625,0.01953125,0.015625,1.0,0.99609375,0.00390625,0.0,0.0,1.0,0.57275390625,0.0,0.0078125,0.0,0.01953125,0.0,0.0,0.039306640625,0.36083984375,1.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.95703125,0.00390625,0.01171875,0.03125,1.0,0.98828125,0.00390625,0.00390625,0.00390625,0.0,1.0],"score":1.0,"idx":5,"name":"hair_confidence","state":True},"gender":{"confidence":[0.99609375,0.00433349609375],"score":0.99609375,"idx":0,"name":"male","state":True},"age":24}
human_dict = {"gender":{"confidence":[0.06760822981595993,0.9323918223381043],"score":0.9323918223381043,"idx":1,"name":"male","state":True},"hair_length":{"confidence":[0.0568091981112957,0.9431908130645752],"score":0.9431908130645752,"idx":1,"name":"short","state":True},"hold_umbrella":{"confidence":[0.9509603977203369,0.04903962463140488],"score":0.9509603977203369,"idx":0,"name":"no","state":True},"orientation":{"confidence":[0.03162207081913948,0.886995255947113,0.058761339634656909,0.022621354088187219],"score":0.886995255947113,"idx":1,"name":"front","state":True},"ride_bike":{"confidence":[0.9485372304916382,0.05146276578307152],"score":0.9485372304916382,"idx":0,"name":"no","state":True},"wear_hat":{"confidence":[0.9435819983482361,0.05641802400350571],"score":0.9435819983482361,"idx":0,"name":"no","state":True},"wear_helmet":{"confidence":[0.9522643685340881,0.047735653817653659],"score":0.9522643685340881,"idx":0,"name":"no","state":True},"carry_any_bag":{"confidence":[0.4868156313896179,0.5131843686103821],"score":0.5131843686103821,"idx":1,"name":"yes","state":True},"carry_backpack":{"confidence":[0.8039445281028748,0.19605544209480287],"score":0.8039445281028748,"idx":0,"name":"no","state":True},"carry_drag":{"confidence":[0.9125920534133911,0.0874079167842865],"score":0.9125920534133911,"idx":0,"name":"no","state":True},"carry_handbag":{"confidence":[0.8688268065452576,0.13117322325706483],"score":0.8688268065452576,"idx":0,"name":"no","state":True},"carry_messengerbag":{"confidence":[0.903037965297699,0.09696201980113983],"score":0.903037965297699,"idx":0,"name":"no","state":True},"dress_lower_size":{"confidence":[0.9472469687461853,0.052752986550331119],"score":0.9472469687461853,"idx":0,"name":"long","state":True},"dress_lower_skirt":{"confidence":[0.948441743850708,0.05155818536877632],"score":0.948441743850708,"idx":0,"name":"no","state":True},"dress_upper_coat":{"confidence":[0.917376697063446,0.08262328803539276],"score":0.917376697063446,"idx":0,"name":"no","state":True},"dress_upper_size":{"confidence":[0.14134065806865693,0.8586593270301819],"score":0.8586593270301819,"idx":1,"name":"short","state":True},"age_num":{"confidence":[0.012984011322259903,0.01676977425813675,0.4070887267589569,0.5046549439430237,0.011949756182730198,0.032865703105926517,0.013687090016901493],"score":0.5046549439430237,"idx":3,"name":"y30_45","state":True}}

hair_list = ["光头", "少量头发（包含秃顶）", "短发", "长发", "无法判断"]
beard_list = ["无胡子或者胡子不明显", "嘴唇上面的胡子", "络腮胡", "无法判断"]
hat_list = ["无帽子", "安全帽", "厨师帽", "学生帽", "头盔", "安防全人种01小白帽", "头巾", "其他帽子", "无法判断"]
respirator_list = ["无口罩", "医用口罩", "雾霾口罩", "普通口罩", "厨房用”透明口罩“", "无法判断"]
glasses_list = ["无眼镜", "深色框透明眼镜", "普通透明眼镜", "墨镜", "无法判断"]
skin_color_list = ["黄皮肤", "白皮肤", "黑皮肤", "褐色皮肤", "无法判断"]
analysis_face_attri_list = [hair_list, beard_list, hat_list, respirator_list, glasses_list, skin_color_list]

analysis_human_attri_list = ["age_num", "carry_any_bag", "carry_backpack", "carry_drag", "carry_handbag", "carry_messengerbag", \
                            "dress_lower_size", "dress_lower_skirt", "dress_upper_coat", "dress_upper_size", "gender", \
                            "hair_length", "hold_umbrella", "ride_bike", "wear_hat", "wear_helmet", "orientation"]

idx_step = 0
for idx in range(len(analysis_face_attri_list)):
    analysis_face_attri = analysis_face_attri_list[idx]

    start_idx = idx_step
    end_idx = start_idx + len(analysis_face_attri)
    attri_list = np.array(face_dict["rc_attr"]['confidence'][start_idx:end_idx])
    attri_name = analysis_face_attri[attri_list.argmax()]

    idx_step += len(analysis_face_attri)
    attri_confidence = face_dict["rc_attr"]['confidence'][idx_step]
    idx_step +=1

    # print(attri_list)
    print(attri_name, attri_confidence)

print("===============================")
for idx in range(len(analysis_human_attri_list)):
    analysis_human_attri = analysis_human_attri_list[idx]
    attri_name = human_dict[analysis_human_attri]["name"]

    print(analysis_human_attri, attri_name)
    

# hair_name = hair_list[np.array(face_dict["rc_attr"]['confidence'][0:5]).argmax()]
# hair_confidence = face_dict["rc_attr"]['confidence'][5]
# print(np.array(face_dict["rc_attr"]['confidence'][0:5]))
# print(hair_name, hair_confidence)

# beard_name = beard_list[np.array(face_dict["rc_attr"]['confidence'][6:10]).argmax()]
# beard_confidence = face_dict["rc_attr"]['confidence'][10]
# print(np.array(face_dict["rc_attr"]['confidence'][6:10]))
# print(beard_name, beard_confidence)

# print(human_dict["orientation"]["name"])