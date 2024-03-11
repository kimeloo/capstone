import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# import
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# 모델 설정
model = RandomForestClassifier()

# CSV 파일 불러오기
pwd = '/Users/kimeloo/Documents/Coding/capstone/data/240201'
data_file = 'test_03_psMatching.csv'
data = pd.read_csv(f'{pwd}/{data_file}')

# 변수 분리
columns = ['bmi_s1', 'age_s1', 'gender', 'htnderv_s1']
X = data[columns].copy()
# X = data.drop(['parrptdiab', 'PropensityScore'], axis=1).copy()
y = data['parrptdiab']

# 데이터 정규화
scaler = StandardScaler()
X_fit = scaler.fit_transform(X)


# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_fit, y, test_size=0.3, random_state=None)


# 모델 설정 및 학습
model.fit(X_train, y_train)

# 모델 평가
test_accuracy = model.score(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# F1 score
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Confusion matrix 시각화
conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
plt.imshow(conf_mat_norm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_mat[i, j]), ha='center', va='center', color='red')
plt.show()
