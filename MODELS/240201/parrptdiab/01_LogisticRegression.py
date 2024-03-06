import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

# CSV 파일 불러오기
pwd = '/Users/kimeloo/Documents/Coding/capstone/data/240201'
train_file = 'diab_train_data_replaced.csv'
test_file = 'shhs1_replaced.csv'
train = pd.read_csv(f'{pwd}/{train_file}')
test = pd.read_csv(f'{pwd}/{train_file}')

# 아웃라이어 제거 전 컬럼
columns = ['diasbp', 'trig', 'savbnbh', 'davbnbh', 'oximet51', 'ai_rem', 'pcs_s1', 'davbnoh', 'savbnoh', 'avgsat']
# 아웃라이어 제거 후 컬럼
columns = ['psg_month', 'oardrop2', 'oardrbp2', 'oarbp2', 'cardnop2', 'canop2', 'cardnbp2', 'canbp2', 'cardrop2', 'carop2']

# 변수 분리
X_train = train[columns].copy()
y_train = train['parrptdiab']
X_test = test[columns].copy()
y_test = test['parrptdiab']

# 데이터 정규화
scaler = StandardScaler()
X_train_fit = scaler.fit_transform(X_train)
X_test_fit = scaler.transform(X_test)

# 모델 학습
model = LogisticRegression()
model.fit(X_train_fit, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test_fit)

# F1 점수 출력
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# confusion matrix 출력
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
plt.xticks(np.arange(2), ['No Diabetes', 'Diabetes'])
plt.yticks(np.arange(2), ['No Diabetes', 'Diabetes'])
plt.show()