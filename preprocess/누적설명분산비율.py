from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data'
filename = 'preprocessed_data.csv'

# DataFrame 생성
df = pd.read_csv(f'{pwd}/{filename}')

# 데이터 표준화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA 모델 생성
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# 누적 설명 분산 비율 계산
cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# 시각화
plt.plot(cumulative_explained_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.show()