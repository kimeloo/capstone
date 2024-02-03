import pandas as pd
# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data'
filename_x = 'PCA_70_data.csv'
filename_y = 'dataset_230122.csv'

# DataFrame 생성
X = pd.read_csv(f'{pwd}/{filename_x}')
y = pd.read_csv(f'{pwd}/{filename_y}')['pd']

# 데이터 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_fit = scaler.fit_transform(X)

# 훈련, 테스트 세트 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_fit, y, test_size=0.2, random_state=100)

################
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns
pca = decomposition.PCA(n_components=1)
sklearn_pca_x = pca.fit_transform(X_train)

sklearn_result = pd.DataFrame(sklearn_pca_x, columns=['PC1'])
sklearn_result['y-axis'] = 0.0
sklearn_result['label'] = y_train
sns.lmplot(x='PC1', y='y-axis', data=sklearn_result, fit_reg=False, scatter_kws={"s": 50}, hue="label")
plt.title('PCA result')
plt.show()