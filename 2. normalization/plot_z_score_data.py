import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# .npy 파일 경로
file_path = "C:/Users/송채영/Desktop/송채영/HAI/code/model/1.npy"

# .npy 파일 로드
data = np.load(file_path)

# 데이터 구조 확인
print("Data shape:", data.shape)

# 데이터 재구성
if data.shape[1] == 29:
    reorganized_data = data[:, :-1]  # 레이블 열을 제외하고 데이터만 추출
else:
    reorganized_data = data

# Z-score 정규화
mean = reorganized_data.mean(axis=1, keepdims=True)
std = reorganized_data.std(axis=1, keepdims=True)
std[std == 0] = 1 
normalized_data = (reorganized_data - mean) / std

print("Mean shape: ", mean.shape)
print("Std shape: ", std.shape)

sample_index = 100 #100, 900, 1700, 2500
original_sample_data = reorganized_data[sample_index, :].reshape(4, 7)
normalized_sample_data = normalized_data[sample_index, :].reshape(4, 7)

plt.figure(figsize=(16, 6))

# 원본 데이터 시각화
plt.subplot(1, 2, 1)
sns.heatmap(original_sample_data, cmap='Blues', annot=True, fmt='.2f', cbar=True, square=True, linewidths=0.5,
            xticklabels=range(original_sample_data.shape[1]), yticklabels=range(original_sample_data.shape[0]))
plt.title(f'Original Data', fontsize=18)

# Z-score 정규화된 데이터 시각화
plt.subplot(1, 2, 2)
sns.heatmap(normalized_sample_data, cmap='Blues', annot=True, fmt='.2f', cbar=True, square=True, linewidths=0.5,
            xticklabels=range(normalized_sample_data.shape[1]), yticklabels=range(normalized_sample_data.shape[0]))
plt.title(f'Z-score Normalized Data', fontsize=18)

for text in plt.gca().texts:
    text.set_size(12)

plt.tight_layout()
plt.show()
