import os
import json

# 获取当前目录下的所有文件名
files = os.listdir()

# 筛选出以"SVM"开头的.json文件
svm_files = [file for file in files if file.startswith('KN') and file.endswith('.json')]
if len(svm_files) == 0:
    print('No files found')
    exit()

for file in svm_files:
    with open(file, 'r') as f:
        data = json.load(f)
        # 检查文件内容中是否包含"mean_test_score"和"std_test_score"这两个关键字
        if 'mean_test_score' in data and 'std_test_score' in data:
            # 将这两个值保留3位小数并打印出来
            print(f'File: {file}')
            print(f'mean_test_score: {data["mean_test_score"]:.3f}')
            print(f'std_test_score: {data["std_test_score"]:.3f}')