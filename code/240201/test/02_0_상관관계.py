import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
pwd = '~/Documents/Coding/capstone/data/240201'
# filename = 'test_01_preprocess.csv'
filename = '04_htnderv_s1_train.csv'
data = pd.read_csv(f'{pwd}/{filename}')

# Calculate the correlation matrix
corr_matrix = data.corr()
top_corr_features = corr_matrix.index[abs(corr_matrix["htnderv_s1"])>=0.1]

# Generate the heatmap
plt.figure(figsize=(13,10))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
# font size 조절
plt.title('Correlation Matrix', fontsize=20)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
# Save the heatmap
plt.show()