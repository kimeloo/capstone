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

# 결측치 비율이 50% 이상인 컬럼 제거
def drop_columns(df):
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio >= 0.5].index
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df

# 0~5 사이 값을 가진 컬럼 : 결측치를 0으로 처리
def fillz(df):
    ### 0~5 사이의 값을 가진 컬럼의 결측치를 0으로 대체
    columns_to_replace = df.columns[(df.min() >= 0) & (df.max() <= 5)].tolist()
    df[columns_to_replace] = df[columns_to_replace].fillna(0)
    return df, columns_to_replace

# 모든 결측치를 평균 값으로 처리
def fill_mean(df, strategy='mean'):
    imputer = SimpleImputer(strategy='mean')
    replaced_data = imputer.fit_transform(df)
    replaced_df = pd.DataFrame(replaced_data, columns=df.columns, index=df.index)
    return replaced_df

def fillz_all(df):
    columns_to_replace = df.columns.tolist()
    df[columns_to_replace] = df[columns_to_replace].fillna(0)
    return df

if __name__ == "__main__":
    # 데이터 불러오기
    pwd = '~/Documents/Coding/capstone/data/240415'
    filename = '02_htnderv_s1_train.csv'
    data = pd.read_csv(f'{pwd}/{filename}')
    data = data.astype('float')

    print(data.shape)

    # 결측치 개수 확인
    print(f'처리 전 결측치 개수: {data.isnull().sum().sum()}')

    # 모델을 사용하여 결측값 대치
    # replaced_data = mice(data)
    data = drop_columns(data)  # 결측치 비율이 50% 이상인 컬럼 제거
    # replaced_data, columns_replaced_z = fillz(data)     # 0~5 사이 값은 0으로, 그 외 값은 평균값 사용
    # replaced_data = fill_mean(replaced_data, strategy='mean')
    replaced_data = fillz_all(data)     # 전체 결측치 0으로 대체

    # 결과 확인
    print(f'처리 후 결측치 개수: {replaced_data.isnull().sum().sum()}')
    print(replaced_data.shape)

    # 결과를 csv 파일로 저장
    replaced_data.to_csv(f'{pwd}/03_htnderv_s1_train.csv', index=False)

    # 결측치 처리 방안 저장(MICE 모델 X)
    filename_original = 'shhs1-dataset-0.20.0.csv'
    original = pd.read_csv(f'{pwd}/{filename_original}', low_memory=False)
    columns_original = original.columns.tolist()
    # print(len(columns_original))
    # print(len(replaced_data.columns.tolist()))
    columns_removed = list(set(columns_original) - set(replaced_data.columns.tolist()))
    
    df = pd.DataFrame(columns=columns_original)
    # 2행에 각 컬럼 평균값 추가
    # df.loc[0] = replaced_data.mean()
    df.loc[0] = 0       # 전체 컬럼 0으로 처리
    # columns_removed에 해당하는 컬럼은 drop으로 처리
    df.loc[0, columns_removed] = 9999999999
    print(len(columns_removed))
    # # columns_replaced_z에 해당하는 컬럼은 0으로 처리
    # df.loc[0, columns_replaced_z] = 0
    # 결측치 처리 방안 저장
    df.to_csv(f'{pwd}/strategy.csv', index=False)