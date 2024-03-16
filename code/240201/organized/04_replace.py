import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor

# 결측치를 다양한 모델로 예측하여 평균값 사용하는 함수 정의
def impute_missing_values(df, num_models=5):
    replaced_data_list = []

    # 모델 리스트 정의
    # models = [RandomForestRegressor(), LogisticRegression(), MLPRegressor(hidden_layer_sizes=(100, 50))]
    models = [MLPRegressor(hidden_layer_sizes=(100, 50))]

    # 모델을 바꿔가며 예측 수행
    for model in models:
        for _ in range(num_models):
            imputer = IterativeImputer(estimator=model, max_iter=100, random_state=2024)
            replaced_data_list.append(imputer.fit_transform(df))
            print(f"Imputing using {model.__class__.__name__} : {replaced_data_list[-1]}")

    # 예측값의 평균 계산
    average_replaced_data = np.mean(replaced_data_list, axis=0)
    # 평균으로 결측치 채우기
    replaced_df = pd.DataFrame(average_replaced_data, columns=df.columns, index=df.index)
    return replaced_df

if __name__ == "__main__":
    # 데이터 불러오기
    pwd = '~/Documents/Coding/capstone/data/240201'
    filename = '02_psMatching_htnderv_s1_train.csv'
    data = pd.read_csv(f'{pwd}/{filename}')

    # 결측치 개수 확인
    print(data.isnull().sum())

    # 모델을 사용하여 결측값 대치
    replaced_data = impute_missing_values(data)

    # 결과 확인
    print(replaced_data.isnull().sum())

    # 결과를 csv 파일로 저장
    replaced_data.to_csv(f'{pwd}/03_replaced_htnderv_s1_train.csv', index=False)