import pandas as pd

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'shhs1-dataset-0.20.0.csv'

# DataFrame 생성 (low_memory=False로 설정하여 데이터 타입을 추론하는 과정을 생략)
df = pd.read_csv(f'{pwd}/{filename}', low_memory=False)
print(df.shape)

# parrptdiab 컬럼의 값이 1인 행만 추출
X = df[df['parrptdiab']==1]
print(X.shape)
