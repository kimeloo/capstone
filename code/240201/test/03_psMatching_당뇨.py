import random
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
# logistic regression 모델 활용
from sklearn.linear_model import LogisticRegression                 # 결과 : 331/331
# discriminant analysis 모델 활용
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis        # 결과 : 331/331
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis     # 결과 : 331/331
# randomforest 모델 활용
from sklearn.ensemble import RandomForestClassifier                # 결과 : 331/331
# boosted cart 모델 활용
from sklearn.ensemble import GradientBoostingClassifier             # 결과 : 331/331

# 활용할 모델 선택
model = LogisticRegression()

# 데이터 불러오기
pwd = '~/documents/coding/capstone/data/240201/'
data = pd.read_csv(pwd + 'test_01_preprocess.csv')

# 데이터 확인
print(data['htnderv_s1'].value_counts())

# Covariate 선택
covariates = ['height', 'weight', 'weight20', 'age_s1', 'gender', 'race']
## Covariate에 과거 질병력 추가
covariates.extend(['srhype', 'cgpkyr', 'alcoh', 'angina15', 'mi15', 'stroke15', 'hf15', 'cabg15', 'ca15', 'othrcs15', 'pacem15', 'sa15', 'emphys15', 'crbron15', 'copd15', 'asthma15', 'asth1215', 'cough315', 'phlegm15', 'runny15', 'sinus15', 'nitro15', 'estrgn1', 'progst1', 'htnmed1', 'anar1a1', 'lipid1', 'sympth1', 'tca1', 'asa1', 'nsaid1', 'benzod1', 'premar1', 'pdei1', 'ntca1', 'warf1', 'loop1', 'hctz1', 'hctzk1', 'ccbir1', 'ccbsr1', 'alpha1', 'alphad1', 'anar1b1', 'anar1c1', 'anar31', 'pvdl1', 'basq1', 'niac1', 'thry1', 'istrd1', 'ostrd1', 'beta1', 'betad1', 'ccb1', 'ace1', 'aced1', 'vaso1', 'vasod1', 'diuret1', 'dig1', 'ntg1', 'htnderv_s1'])

# Logistic Regression을 사용하여 Propensity Score 계산
X = data[covariates]
y = data['htnderv_s1']
model.fit(X, y)
data['PropensityScore'] = model.predict_proba(X)[:, 1]

# htnderv_s1 그룹과 Control 그룹 간의 유사한 개체를 매칭
htnderv_s1_group = data[data['htnderv_s1'] == 1]
control_group = data[data['htnderv_s1'] == 0]

X_htnderv_s1 = htnderv_s1_group['PropensityScore'].values.reshape(-1, 1)
X_control = control_group['PropensityScore'].values.reshape(-1, 1)

# Nearest-neighbor 매칭
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_control)
distances, indices = nn.kneighbors(X_htnderv_s1)
matched_control_indices = indices.flatten()
matched_control_group = control_group.iloc[matched_control_indices]

# 매칭된 데이터셋 생성
matched_data = pd.concat([htnderv_s1_group, matched_control_group])

# 결과 확인
# print(matched_data.head())

# htnderv_s1가 0인 데이터와 1인 데이터의 개수 확인
print(matched_data['htnderv_s1'].value_counts())

# csv 파일로 저장
matched_data.to_csv(pwd + 'test_03_psMatching.csv', index=False)