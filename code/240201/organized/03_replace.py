from os import replace
import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
import joblib

# 결측치를 MICE로 처리
def mice(df):
    # DNN 모델 생성
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=2024, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=10)
    # IterativeImputer(MICE) 사용
    imputer = IterativeImputer(estimator=model, verbose=2)
    replaced_data = imputer.fit_transform(df)
    
    # 모델 저장
    print("Dumping model...")
    joblib.dump(imputer, 'dnn_imputer.pkl')
    
    # 결과 데이터프레임 생성
    replaced_df = pd.DataFrame(replaced_data, columns=df.columns, index=df.index)
    return replaced_df

# 결측치를 특정 값으로 처리
def simple(df, strategy='mean'):
    imputer = SimpleImputer(strategy='mean')
    replaced_data = imputer.fit_transform(df)
    replaced_df = pd.DataFrame(replaced_data, columns=df.columns, index=df.index)
    return replaced_df

if __name__ == "__main__":
    # 데이터 불러오기
    pwd = '~/Documents/Coding/capstone/data/240201'
    filename = '02_htnderv_s1_train.csv'
    data = pd.read_csv(f'{pwd}/{filename}')

    # 결측치 개수 확인
    print(data.isnull().sum())

    # 모델을 사용하여 결측값 대치
    # replaced_data = mice(data)
    replaced_data = simple(data, strategy='mean')      # 평균값 사용
    # replaced_data = simple(data, strategy='most_frequent')      # 최빈값 사용

    # 결과 확인
    print(replaced_data.isnull().sum())

    # 결과를 csv 파일로 저장
    replaced_data.to_csv(f'{pwd}/03_htnderv_s1_train.csv', index=False)
