import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'preprocessed_htnderv_s1.csv'
data = pd.read_csv(f'{pwd}/{filename}')

# Covariate 선택
covariates = ['height', 'weight', 'weight20', 'age_s1', 'gender', 'race']
# Covariate에 과거 질병력 추가
covariates.extend(['srhype', 'cgpkyr', 'alcoh', 'angina15', 'mi15', 'stroke15', 'hf15', 'cabg15', 'ca15', 'othrcs15', 'pacem15', 'sa15', 'emphys15', 'crbron15', 'copd15', 'asthma15', 'asth1215', 'cough315', 'phlegm15', 'runny15', 'sinus15', 'nitro15', 'estrgn1', 'progst1', 'htnmed1', 'anar1a1', 'lipid1', 'sympth1', 'tca1', 'asa1', 'nsaid1', 'benzod1', 'premar1', 'pdei1', 'ntca1', 'warf1', 'loop1', 'hctz1', 'hctzk1', 'ccbir1', 'ccbsr1', 'alpha1', 'alphad1', 'anar1b1', 'anar1c1', 'anar31', 'pvdl1', 'basq1', 'niac1', 'thry1', 'istrd1', 'ostrd1', 'beta1', 'betad1', 'ccb1', 'ace1', 'aced1', 'vaso1', 'vasod1', 'diuret1', 'dig1', 'ntg1'])

# htnderv_s1 컬럼이 0과 1인 그룹 각각의 height 컬럼 값의 평균과 표준편차 계산
group0 = data[data['htnderv_s1'] == 0]['height']
group1 = data[data['htnderv_s1'] == 1]['height']
print(f'htnderv_s1이 0인 그룹의 height 평균: {group0.mean()}, 표준편차: {group0.std()}')
print(f'htnderv_s1이 1인 그룹의 height 평균: {group1.mean()}, 표준편차: {group1.std()}')

# covariates에 저장된 변수들을 이용하여, htnderv_s1컬럼이 0과 1인 그룹 각각의 T-score와 p-value를 계산
for covariate in covariates:
    group0 = data[data['htnderv_s1'] == 0][covariate]
    group1 = data[data['htnderv_s1'] == 1][covariate]
    t_score, p_value = stats.ttest_ind(group0, group1)
    print(f'{covariate}의 T-score: {t_score}, p-value: {p_value}')

# T-score와 p-value를 csv 파일로 저장
t_score_p_value = pd.DataFrame({'covariate': [], 't_score': [], 'p_value': []})
for covariate in covariates:
    group0 = data[data['htnderv_s1'] == 0][covariate]
    group1 = data[data['htnderv_s1'] == 1][covariate]
    t_score, p_value = stats.ttest_ind(group0, group1)
    # 과학적 표기법으로 표현된 p-value를 일반적인 숫자로 변환
    p_value = '%.10f' % p_value
    t_score_p_value = pd.concat([t_score_p_value, pd.DataFrame({'covariate': [covariate], 't_score': [t_score], 'p_value': [p_value]})], ignore_index=True)
t_score_p_value.to_csv(f'{pwd}/t_score_p_value.csv', index=False)
