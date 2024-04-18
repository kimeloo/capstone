import xgboost as xgb
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

# XGBoost 데이터셋으로 변환
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

# XGBoost 모델 파라미터 설정
# accuracy : 0.9558
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'max_depth': 6,
    'eta': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# 모델 훈련
num_round = 100
bst = xgb.train(params, train_data, num_round, evals=[(test_data, 'test')])

# 테스트 데이터에 대한 예측
y_pred = bst.predict(test_data)

# 예측값을 이진 클래스로 변환
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'XGBoost Accuracy: {accuracy}')
