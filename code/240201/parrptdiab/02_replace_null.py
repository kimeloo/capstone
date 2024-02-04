import pwd
import pandas as pd
from sqlalchemy import column

def replace_missing_values(df):
    # 모든 컬럼을 int 또는 float로 변환
    df = df.astype('float')

    # 각 컬럼의 결측치 비율 계산
    missing_ratio = df.isnull().sum() / len(df)
    
    # 결측치 비율이 50% 이상인 컬럼 제거
    columns_to_drop = missing_ratio[missing_ratio >= 0.5].index
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    # 0~5 사이의 값을 가진 컬럼의 결측치를 0으로 대체
    columns_to_replace = df.columns[(df.min() >= 0) & (df.max() <= 5)].tolist()
    df[columns_to_replace] = df[columns_to_replace].fillna(0)
    
    # 그 이외의 값을 가진 컬럼의 결측치를 평균으로 대체
    columns_to_replace = df.columns[(df.min() < 0) | (df.max() > 5)].tolist()
    df[columns_to_replace] = df[columns_to_replace].fillna(df.mean())
    
    return df

# CSV 파일 불러오기
pwd = '/Users/kimeloo/Documents/Coding/capstone/data/240201'
file_name = 'diab_train_data.csv'
df = pd.read_csv(f'{pwd}/{file_name}')

# 필요 없는 컬럼 제거
## nsrrid, pptid, pptidr : ID
df = df.drop(columns=['nsrrid', 'pptid', 'pptidr'])

# 시간 데이터 정수로 변환
column_name = 'rcrdtime'
change_name = column_name + "_numeric"
df[column_name] = pd.to_datetime(df[column_name], format='%H:%M:%S', errors='coerce').dt.time
df[change_name] = [(t.hour*3600 + t.minute*60 + t.second) if not pd.isna(t) else pd.NaT for t in df[column_name]]
df[column_name] = df[change_name]
df = df.drop(columns=change_name)

# 결측치 처리 함수 호출
df = replace_missing_values(df)

# 결과 저장
df.to_csv(f'{pwd}/diab_train_replaced.csv', index=False)