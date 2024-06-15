import os
import json
import numpy as np

# 获取当前目录下的所有文件名
files = os.listdir()

# 筛选出.json文件
json_files = [file for file in files if file.endswith('.json')]

for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        # 计算列表的平均值
        mean_value = np.mean(data)
        print(f'File: {file}, Mean: {mean_value}')