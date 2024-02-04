import pandas as pd

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'shhs1-dataset-0.20.0.csv'

# DataFrame 생성
df = pd.read_csv(f'{pwd}/{filename}')

# # 입력 데이터와 결과 분리
# X = df.drop('bpmeds', axis=1)
# y = df['bpmeds']

# bpmeds 컬럼의 값이 1인 행만 추출
X = df[df['bpmeds']==1]
print(X.shape)
## bpmeds 컬럼이 없어 결과 추출 불가