import pandas as pd

# # 파일 경로, 파일명 입력
# pwd = '~/Documents/Coding/capstone/data/240201'
# filename = 'shhs1_original.csv'

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240122'
filename = 'dataset_230122.csv'

# DataFrame 생성
df = pd.read_csv(f'{pwd}/{filename}')
print(df.head())
print(df.shape)