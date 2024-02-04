import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'diab_train_data_replaced.csv'

# DataFrame 생성
X = pd.read_csv(f'{pwd}/{filename}').drop(columns='parrptdiab')
y = pd.read_csv(f'{pwd}/{filename}')['parrptdiab']

# 데이터 정규화
scaler = StandardScaler()
X_fit = scaler.fit_transform(X)

# 누적된 중요도 저장할 배열
cumulative_importance = np.zeros(X.shape[1])

# 시행 횟수
num_trials = 100

# train_test_split을 여러 번 시행하여 중요도 누적
for _ in range(num_trials):
    X_train, X_test, y_train, y_test = train_test_split(X_fit, y, test_size=0.2, random_state=None) # random_state=None으로 설정하여 매번 다른 데이터셋 생성
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    feature_importances = rf_model.feature_importances_
    cumulative_importance += feature_importances

# 누적 중요도 기준으로 상위 10개의 변수 선택
sorted_indices = np.argsort(cumulative_importance)[::-1]
top_10_indices = sorted_indices[:10]
top_10_features = X.columns[top_10_indices]

# 선택된 상위 10개 변수 출력
print("Top 10 Features:")
for i, feature in enumerate(top_10_features, 1):
    print(f"{i}. {feature}")
