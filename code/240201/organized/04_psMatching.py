import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
# logistic regression 모델 활용
from sklearn.linear_model import LogisticRegression
# discriminant analysis 모델 활용
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# randomforest 모델 활용
from sklearn.ensemble import RandomForestRegressor
# boosted cart 모델 활용
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

class psMatching:
    def __init__(self, data):
        self.data = data
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'verbosity': -1
        }
        # self.model = lgb.LGBMClassifier(**self.params)
        self.model = LogisticRegression(max_iter=2000, random_state=2024)
        self.covariates = ['height', 'weight', 'weight20', 'age_s1', 'gender', 'race']
        # self.covariates.extend(['srhype', 'cgpkyr', 'alcoh', 'angina15', 'mi15', 'stroke15', 'hf15', 'cabg15', 'ca15', 'othrcs15', 'pacem15', 'sa15', 'emphys15', 'crbron15', 'copd15', 'asthma15', 'asth1215', 'cough315', 'phlegm15', 'runny15', 'sinus15', 'nitro15', 'estrgn1', 'progst1', 'htnmed1', 'anar1a1', 'lipid1', 'sympth1', 'tca1', 'asa1', 'nsaid1', 'benzod1', 'premar1', 'pdei1', 'ntca1', 'warf1', 'loop1', 'hctz1', 'hctzk1', 'ccbir1', 'ccbsr1', 'alpha1', 'alphad1', 'anar1b1', 'anar1c1', 'anar31', 'pvdl1', 'basq1', 'niac1', 'thry1', 'istrd1', 'ostrd1', 'beta1', 'betad1', 'ccb1', 'ace1', 'aced1', 'vaso1', 'vasod1', 'diuret1', 'dig1', 'ntg1', 'htnderv_s1'])
    
    def psMatching(self):
        # Propensity Score 계산
        X = self.data[self.covariates]
        y = self.data['htnderv_s1']
        self.model.fit(X, y)

        self.data['PropensityScore'] = self.model.predict_proba(X)[:, 1]
        # self.data['PropensityScore'] = self.model.predict(X)      # lightgbm의 경우 predict_proba가 없음

        # htnderv_s1 그룹과 Control 그룹 지정
        htnderv_s1_group = self.data[self.data['htnderv_s1'] == 1]
        control_group = self.data[self.data['htnderv_s1'] == 0]

        X_htnderv_s1 = htnderv_s1_group['PropensityScore'].values.reshape(-1, 1)
        X_control = control_group['PropensityScore'].values.reshape(-1, 1)

        # Nearest-neighbor 매칭
        matched_control_group = self.nnMatching(X_control, X_htnderv_s1, control_group)
                
        # 매칭된 데이터셋 생성
        matched_train = pd.concat([htnderv_s1_group, matched_control_group])

        # 중복 제거
        matched_train.drop_duplicates(inplace=True)
        
        return matched_train
                

    def nnMatching(self, X_control, X_htnderv_s1, control_group):
        # Nearest-neighbor 매칭
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_control)
        _, indices = nn.kneighbors(X_htnderv_s1)
        matched_control_indices = indices.flatten()
        matched_control_group = control_group.iloc[matched_control_indices]

        return matched_control_group

        


if __name__ == "__main__":
    # 데이터 불러오기
    pwd = '~/documents/coding/capstone/data/240201/'
    train_data = pd.read_csv(pwd + '03_htnderv_s1_train.csv')
    # test_data = pd.read_csv(pwd + '03_htnderv_s1_test.csv')

    # 데이터 확인
    print(train_data['htnderv_s1'].value_counts())

    # psMatching
    match_train = psMatching(train_data)
    matched_train = match_train.psMatching()

    # 중복 개수 확인
    print(matched_train.duplicated().sum())

    # 결과 확인
    print(matched_train.tail())

    # htnderv_s1가 0인 데이터와 1인 데이터의 개수 확인
    print(matched_train['htnderv_s1'].value_counts())

    # 전체 데이터를 csv 파일로 저장
    matched_train.to_csv(pwd + '04_htnderv_s1_all.csv', index=False)

    # train-val set split
    from sklearn.model_selection import train_test_split
    X = matched_train
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=2024)

    # 각각 저장
    X_train.to_csv(pwd + '04_htnderv_s1_train.csv', index=False)
    X_val.to_csv(pwd + '04_htnderv_s1_val.csv', index=False)
    # X_test.to_csv(pwd + '04_htnderv_s1_test.csv', index=False)
