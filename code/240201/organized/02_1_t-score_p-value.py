import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate(data, covariates):
    # T-score와 p-value 계산
    t_score_p_value = pd.DataFrame({'covariate': [], 't_score': [], 'p_value': []})
    for covariate in covariates:
        group0 = data[data['htnderv_s1'] == 0][covariate]
        group1 = data[data['htnderv_s1'] == 1][covariate]
        t_score, p_value = stats.ttest_ind(group0, group1)
        # p-value를 일반적인 숫자로 변환
        p_value = '%.10f' % p_value
        t_score_p_value = pd.concat([t_score_p_value, pd.DataFrame({'covariate': [covariate], 't_score': [t_score], 'p_value': [p_value]})], ignore_index=True)
    return t_score_p_value

if __name__ == "__main__":
    # 파일 경로, 파일명 입력
    pwd = '~/Documents/Coding/capstone/data/240201'
    filename_before = '01_htnderv_s1.csv'
    filename_after = '02_psMatching_htnderv_s1_all.csv'
    data_before = pd.read_csv(f'{pwd}/{filename_before}')
    data_after = pd.read_csv(f'{pwd}/{filename_after}')

    # Covariate 선택
    covariates = ['height', 'weight', 'weight20', 'age_s1', 'gender', 'race']
    # Covariate에 과거 질병력 추가
    covariates.extend(['srhype', 'cgpkyr', 'alcoh', 'angina15', 'mi15', 'stroke15', 'hf15', 'cabg15', 'ca15', 'othrcs15', 'pacem15', 'sa15', 'emphys15', 'crbron15', 'copd15', 'asthma15', 'asth1215', 'cough315', 'phlegm15', 'runny15', 'sinus15', 'nitro15', 'estrgn1', 'progst1', 'htnmed1', 'anar1a1', 'lipid1', 'sympth1', 'tca1', 'asa1', 'nsaid1', 'benzod1', 'premar1', 'pdei1', 'ntca1', 'warf1', 'loop1', 'hctz1', 'hctzk1', 'ccbir1', 'ccbsr1', 'alpha1', 'alphad1', 'anar1b1', 'anar1c1', 'anar31', 'pvdl1', 'basq1', 'niac1', 'thry1', 'istrd1', 'ostrd1', 'beta1', 'betad1', 'ccb1', 'ace1', 'aced1', 'vaso1', 'vasod1', 'diuret1', 'dig1', 'ntg1'])

    # T-score와 p-value 계산
    t_score_p_value_before = calculate(data_before, covariates)
    t_score_p_value_after = calculate(data_after, covariates)
    t_score_p_value_before.to_csv(f'{pwd}/02_1_t_score_p_value_before.csv', index=False)
    t_score_p_value_after.to_csv(f'{pwd}/02_1_t_score_p_value_after.csv', index=False)

    # T-score, p-value 변화 확인
    dataframes = [t_score_p_value_before, t_score_p_value_after]
    for df in dataframes:
        df.set_index('covariate', inplace=True)
        df['t_score'] = df['t_score'].astype(float)
        df['p_value'] = df['p_value'].astype(float)
    t_score_p_value_before['t_score'].plot(kind='bar', color='blue', alpha=0.7, label='Before')
    t_score_p_value_after['t_score'].plot(kind='bar', color='red', alpha=0.7, label='After')
    plt.legend()
    plt.title('T-score 변화')
    plt.savefig(f'{pwd}/02_1_t_score.png')
    plt.show()
    t_score_p_value_before['p_value'].plot(kind='bar', color='blue', alpha=0.7, label='Before')
    t_score_p_value_after['p_value'].plot(kind='bar', color='red', alpha=0.7, label='After')
    plt.legend()
    plt.title('p-value 변화')
    plt.savefig(f'{pwd}/02_1_p_value.png')
    plt.show()

