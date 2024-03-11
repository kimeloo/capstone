import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'test_03_psMatching_tca1.csv'

# DataFrame 생성
X = pd.read_csv(f'{pwd}/{filename}').drop(['tca1', 'PropensityScore'], axis=1)
y = pd.read_csv(f'{pwd}/{filename}')['tca1']

columns = []
for i in ['av', 'mn', 'mx']:
    columns.append(f'{i}hrbp')
    columns.append(f'{i}hrop')
    columns.append(f'{i}hnbp')
    columns.append(f'{i}hnop')
X = X[columns].copy()

# 데이터 정규화
scaler = StandardScaler()
X_fit = scaler.fit_transform(X)

# 훈련, 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X_fit, y, test_size=0.2, random_state=None)

# 랜덤 포레스트 분류기 생성
rf_model = RandomForestClassifier()

# 모델 학습
rf_model.fit(X_train, y_train)

# 변수 중요도 확인
feature_importances = rf_model.feature_importances_

# 변수 중요도를 기준으로 내림차순
sorted_indices = np.argsort(feature_importances)[::-1]

# 상위 n개의 변수 선택
n_selected_features = 100
selected_features_indices = sorted_indices[:n_selected_features]

# 선택된 변수의 인덱스 출력
selected_features_names = [f"{index+1}" for index in selected_features_indices]
print("Selected Features:", selected_features_names)

# 변수 중요도가 0.01 이상인 변수들의 인덱스 선택
significant_features_indices = sorted_indices[feature_importances[sorted_indices] >= 0.01]

# 선택된 변수의 이름을 CSV 파일로 저장
selected_features_df = pd.DataFrame({"Variable_Name": X.columns[selected_features_indices]})
# selected_features_df.to_csv(f'{pwd}/selected_features.csv', index=False)

# 중요도가 0.01 이상인 변수들의 중요도 및 이름 저장 리스트
significant_features_importance = []
significant_features_names = []

# 중요도가 0.01 이상인 변수들의 중요도 및 이름 저장
for index in significant_features_indices:
    significant_features_importance.append(feature_importances[index])
    significant_features_names.append(X.columns[index])

# 그래프
plt.figure(figsize=(12, 6))
sns.barplot(x=significant_features_importance, y=significant_features_names, palette="viridis")
plt.title("Feature Importance of Significant Variables")
plt.xlabel("Feature Importance")
plt.ylabel("Variable Name")
plt.show()
