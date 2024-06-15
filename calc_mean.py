import json
import numpy as np

# 读取kfold_results_acc.json文件的内容
with open('kfold_results_acc.json', 'r') as f:
    data = json.load(f)

# 遍历字典
for key, values in data.items():
    # 计算列表的平均值
    mean_value = np.mean(values)
    print(f'Key: {key}, Mean: {mean_value}')
    
import matplotlib.pyplot as plt
import matplotlib.style as style

# 使用ggplot风格
style.use('ggplot')

# 创建一个新的figure和axes
fig, ax = plt.subplots()

# 获取字典的关键词和对应的平均值
keys = data.keys()
means = [np.mean(values) for values in data.values()]

# 绘制柱状图，设置柱状图的颜色为深蓝色
bars = ax.bar(keys, means, color='darkblue')

# 在每个柱子的顶部添加文本
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "{:.3f}".format(mean), 
            ha='center', va='bottom')

# 设置x轴和y轴的标签
ax.set_xlabel('Classifiers')
ax.set_ylabel('Mean Accuracy')

# 显示图表
plt.show()