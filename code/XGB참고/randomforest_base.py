from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from variable_names import columns_to_model

# CSV 파일 불러오기
data = pd.read_csv('train.csv')  # CSV 파일 경로 지정

# 특성과 타겟 변수를 분리합니다.
selected_columns = columns_to_model
X = data[selected_columns]
y = data['htnderv_s1']

# 데이터를 훈련 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest 모델 생성
# accuracy : 0.9272
rf_model = RandomForestClassifier(random_state=42)

# 모델 훈련
rf_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred_rf = rf_model.predict(X_test)

# 정확도 평가
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'RandomForest Accuracy: {accuracy_rf}')
