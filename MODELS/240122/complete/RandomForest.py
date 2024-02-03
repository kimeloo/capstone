from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240122'
filename_x = 'preprocessed_data.csv'
filename_y = 'dataset_230122.csv'

# DataFrame 생성
X = pd.read_csv(f'{pwd}/{filename_x}')
y = pd.read_csv(f'{pwd}/{filename_y}')['pd']
X = X[['cvtia', 'cvcer', 'poremlat', 'potmrem', 'pqpenth', 'poremli', 'm1adepr', 'poremlii', 'fosofam1', 'poplmrem']].copy()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 랜덤 포레스트 모델 생성
model = RandomForestClassifier()

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy :", accuracy)
print("F1-Score :", f1)

# 분류 보고서 출력
print(classification_report(y_test, y_pred))

# confusion matrix 계산
cm = confusion_matrix(y_test, y_pred)

# confusion matrix 그래프 출력
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})  # annot_kws를 사용하여 텍스트 크기 조절
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('pd')
plt.show()
