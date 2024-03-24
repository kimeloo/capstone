import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 불러오기
pwd = '~/documents/coding/capstone/data/240314/'
data = pd.read_csv(pwd + '01_2_htnderv_s1.csv')

# train-val-test set split
# train:test = 7:3
train, test = train_test_split(data, test_size=0.3, random_state=2024)
# train:val = 8:2
train, val = train_test_split(train, test_size=0.2, random_state=2024)

# 데이터 확인
print(f'original: {data.shape}')
print(f'train: {train.shape}')
print(f'val: {val.shape}')
print(f'test: {test.shape}')

# 각각 저장
train.to_csv(pwd + '02_htnderv_s1_train.csv', index=False)
val.to_csv(pwd + '02_htnderv_s1_val.csv', index=False)
test.to_csv(pwd + '02_htnderv_s1_test.csv', index=False)