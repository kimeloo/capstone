import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'test_01_preprocess.csv'
data = pd.read_csv(f'{pwd}/{filename}')

# Calculate the correlation matrix
corr_matrix = data.corr()
top_corr_features = corr_matrix.index[abs(corr_matrix["parrptdiab"])>=0.1]

# Generate the heatmap
plt.figure(figsize=(13,10))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()