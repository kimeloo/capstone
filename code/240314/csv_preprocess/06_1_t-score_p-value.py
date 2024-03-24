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
        # 과학적 표기법을 float으로 변환
        t_score = float(f'{t_score:.10f}')
        p_value = float(f'{p_value:.10f}')
        t_score_p_value = pd.concat([t_score_p_value, pd.DataFrame({'covariate': [covariate], 't_score': [t_score], 'p_value': [p_value]})], ignore_index=True)
    return t_score_p_value

if __name__ == "__main__":
    # 파일 경로, 파일명 입력
    pwd = '~/Documents/Coding/capstone/data/240314'
    filename_before = '03_htnderv_s1_train.csv'
    filename_after = '04_htnderv_s1_train.csv'
    data_before = pd.read_csv(f'{pwd}/{filename_before}')
    data_after = pd.read_csv(f'{pwd}/{filename_after}')

    # Covariate 선택
    covariates = ['bmi_s1', 'age_s1', 'gender']
    
    # T-score와 p-value 계산
    t_score_p_value_before = calculate(data_before, covariates)
    t_score_p_value_after = calculate(data_after, covariates)
    t_score_p_value_before.to_csv(f'{pwd}/05_1_t_score_p_value_before.csv', index=False)
    t_score_p_value_after.to_csv(f'{pwd}/05_1_t_score_p_value_after.csv', index=False)

    # T-score, p-value 변화 확인
    dataframes = [t_score_p_value_before, t_score_p_value_after]
    for df in dataframes:
        df.set_index('covariate', inplace=True)
        df['t_score'] = df['t_score'].astype(float)
        df['p_value'] = df['p_value'].astype(float)
        print(df['p_value'])
    t_score_p_value_before['t_score'].plot(kind='bar', color='blue', label='Before')
    t_score_p_value_after['t_score'].plot(kind='bar', color='red', label='After')
    plt.legend()
    plt.title('T-score change')
    # plt.savefig(f'{pwd}/02_1_t_score.png')
    plt.show()
    t_score_p_value_before['p_value'].plot(kind='bar', color='blue', label='Before', position=1, width=0.4)
    t_score_p_value_after['p_value'].plot(kind='bar', color='red', label='After', position=0, width=0.4)
    plt.legend()
    plt.title('p-value change')
    # plt.savefig(f'{pwd}/02_1_p_value.png')
    plt.show()

