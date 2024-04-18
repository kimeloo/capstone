import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pandas as pd

# 다른 파일 불러오기
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dict import dict_list

# CSV 파일 불러오기
train_data = pd.read_csv('final_data/train.csv')  # 훈련 데이터 파일 경로 지정
validation_data = pd.read_csv('final_data/val.csv')  # 검증 데이터 파일 경로 지정

# X = data.drop(['htnderv_s1', 'nsrrid'], axis=1)
# y = data['htnderv_s1']

# 특성과 타겟 변수를 분리합니다.

list_columns = dict_list.general_health
selected_columns = list_columns

X_train = train_data[selected_columns]
y_train = train_data['htnderv_s1']

X_val = validation_data[selected_columns]
y_val = validation_data['htnderv_s1']

# LightGBM 데이터셋으로 변환
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_val, label=y_val)

# LightGBM 모델 파라미터 설정
# accuracy : 0.9643
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 모델 훈련
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[validation_data])

# 검증 데이터에 대한 예측
y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)

# 예측값을 이진 클래스로 변환
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# 정확도 평가
accuracy = accuracy_score(y_val, y_pred_binary)
print(f'Validation Accuracy: {accuracy}')
