# 라이브러리 불러오기
import pandas as pd
import xgboost as xgb

# 데이터 불러오기
pwd = '/users/kimeloo/Documents/Coding/capstone/data/240314'
df = pd.read_csv(pwd+'/train.csv')

# 특성과 목표 변수 분리
X = df.drop(columns=['htnderv_s1', 'ccb1', 'beta1', 'ace1', 'diuret1', 'nsrrid', 'ccbsr1'])
y = df['htnderv_s1']

# XGBoost 모델 학습
model = xgb.XGBClassifier(n_estimators=100)
model.fit(X, y)

# Weight 기준 Feature Importance 추출
weight_importance = model.get_booster().get_score(importance_type='weight')

# Cover 기준 Feature Importance 추출
cover_importance = model.get_booster().get_score(importance_type='cover')

# Gain 기준 Feature Importance 추출
gain_importance = model.get_booster().get_score(importance_type='gain')

# 결과 출력
for f in [weight_importance, cover_importance, gain_importance]:
    sorted_importance = sorted(f.items(), key=lambda x: x[1], reverse=True)
    print("[", end="")
    for i in range(30):
        print(f"'{sorted_importance[i][0]}'", end=", ")
    print("]")
