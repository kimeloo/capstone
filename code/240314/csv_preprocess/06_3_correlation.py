import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
pwd = '~/Documents/Coding/capstone/data/240201'
filename = '04_htnderv_s1_train.csv'
data = pd.read_csv(f'{pwd}/{filename}')

# 상관관계 계산
corr_matrix = data.corr()
top_corr_features = corr_matrix.index[abs(corr_matrix["htnderv_s1"])>=0.1]

# heatmap 생성
plt.figure(figsize=(13,10))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, annot_kws={"size": 5}, cmap="RdYlGn")
pwd = '/Users/kimeloo'+pwd[1:]
plt.savefig(f'{pwd}/06_3_correlation_heatmap.png')
plt.show()
