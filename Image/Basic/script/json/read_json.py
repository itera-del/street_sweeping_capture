import json

# 指定JSON文件的路径
file_path = '/yuanhuan/data/image/temp/status_caches.json'

# 打开JSON文件，并读取数据
with open(file_path, 'r', encoding='utf-8') as file:
    # 使用json.load()函数将文件内容解析为Python字典
    data = json.load(file)

# # 现在data变量包含了JSON文件中的数据
# print(data)

# 你可以像操作普通字典一样操作data
print()