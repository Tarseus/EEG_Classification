import scipy.io
import matplotlib.pyplot as plt

# 读取.mat文件
data_dir = './data/'
mat = scipy.io.loadmat(data_dir + 's1.mat')

# 提取第一个EEG信号
eeg_signal = mat['X'][0]

# 创建图像
fig, axs = plt.subplots(59, 1, figsize=(10, 6))
for i in range(59):
    axs[i].plot(eeg_signal[i])
    axs[i].axis('off')
    axs[i].text(-0.05, 0.5, f'Channel {i+1}', va='center', ha='right', transform=axs[i].transAxes)
plt.tight_layout()
plt.show()