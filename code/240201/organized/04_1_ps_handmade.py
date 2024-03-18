import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class psMatching:
    def __init__(self, data):
        self.data = data
        self.model = LogisticRegression(max_iter=2000, random_state=2024)
        self.covariates = ['height', 'weight', 'weight20', 'age_s1', 'gender', 'race']
    
    def calculate(self):
        # Propensity Score 계산 모델 학습
        X = self.data[self.covariates]
        y = self.data['htnderv_s1']
        self.model.fit(X,y)

        # Propensity Score 계산
        self.data['PropensityScore'] = self.model.predict_proba(X)[:, 1]

        # htnderv_s1 그룹과 Control 그룹 지정
        htnderv_s1_group = self.data[self.data['htnderv_s1'] == 1]
        control_group = self.data[self.data['htnderv_s1'] == 0]

        # Propensity Score 추출
        X_htnderv_s1 = htnderv_s1_group['PropensityScore'].values.reshape(-1, 1)
        X_control = control_group['PropensityScore'].values.reshape(-1, 1)
    
        return htnderv_s1_group, control_group, X_htnderv_s1, X_control
    
    def nearest_match(self, X_htnderv_s1_each, X_control_list):
        '''
        X_htnderv_s1 그룹 중 하나의 데이터에 대해 가장 가까운 Control 그룹 데이터 찾기
        nearest-neighbor 매칭 방식이지만 중복을 원천차단하기 위함
        X_control_list를 함수 실행마다 업데이트 해야 함
        '''
        # 각 Control 그룹 데이터와의 거리 계산
        distance = np.abs(X_htnderv_s1_each - X_control_list)
        # 가장 가까운 데이터의 인덱스 반환
        nearest_index = np.argmin(distance)
        
        return nearest_index

    def match(self):
        htnderv_s1_group, control_group, X_htnderv_s1, X_control = self.calculate()
        matched_control_group = pd.DataFrame()
        # Nearest-neighbor 매칭
        for idx, X_htnderv_s1_each in enumerate(X_htnderv_s1):
            # 가장 가까운 Control 그룹 데이터 Index 찾기
            nearest_index = self.nearest_match(X_htnderv_s1_each, X_control)
            print(f'{idx:>5}st nearest_index: {nearest_index}')
            # 해당 Index에 위치한 X_control 행 제거
            # print(f'X_control: \n{X_control[nearest_index]}')
            X_control = np.delete(X_control, nearest_index)
            
            # 해당 Index에 위치한 control_group 행 데이터 matched_control_group에 추가 및 제거
            # matched_control_group = pd.concat([matched_control_group, control_group.iloc[nearest_index]])
            # new_row = control_group.iloc[nearest_index]  # 추가할 행의 데이터
            # matched_control_group.loc[len(matched_control_group)] = new_row
            if matched_control_group.empty:
                matched_control_group = pd.DataFrame(columns=control_group.columns)
            matched_control_group.loc[len(matched_control_group)] = control_group.iloc[nearest_index]


            # print(f'matched_control_group: \n{control_group.iloc[nearest_index]}\n{control_group.index[nearest_index]}')
            control_group = control_group.drop(control_group.index[nearest_index])
            # print(f'control_group: \n{control_group}')
            # input()
        print(f'\nfinal matched_control_group : \n{matched_control_group}')
        # 매칭된 데이터셋 생성
        matched_train = pd.concat([htnderv_s1_group, matched_control_group])

        return matched_train

if __name__ == "__main__":
    # 데이터 불러오기
    pwd = '~/documents/coding/capstone/data/240201/'
    train_data = pd.read_csv(pwd + '03_htnderv_s1_train.csv')
    # test_data = pd.read_csv(pwd + '03_htnderv_s1_test.csv')

    # psMatching
    match_train = psMatching(train_data)
    matched_train = match_train.match()

    # 중복 개수 확인
    print(f'duplicated : {matched_train.duplicated().sum()}')

    # 데이터 확인
    print(f"before : {train_data['htnderv_s1'].value_counts()}")
    print(f"after : {matched_train['htnderv_s1'].value_counts()}")
    # 전체 데이터를 csv 파일로 저장
    matched_train.to_csv(pwd + '04_htnderv_s1_all.csv', index=False)