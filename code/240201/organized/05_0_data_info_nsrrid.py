import pandas as pd

# 파일 불러오기
pwd = '~/documents/coding/capstone/data/240201'
train_csv = pd.read_csv(pwd + '/04_htnderv_s1_train.csv')
val_csv = pd.read_csv(pwd + '/02_htnderv_s1_val.csv')
test_csv = pd.read_csv(pwd + '/02_htnderv_s1_test.csv')
unmatched_csv = pd.read_csv(pwd + '/04_htnderv_s1_unmatched.csv')

# 데이터 확인
print(f'train_csv: {train_csv.shape}')
print(f'val_csv: {val_csv.shape}')
print(f'test_csv: {test_csv.shape}')
print(f'unmatched_csv: {unmatched_csv.shape}')

# 파일별 nsrrid 추출
train = train_csv['nsrrid']
val = val_csv['nsrrid']
test = test_csv['nsrrid']
unmatched = unmatched_csv['nsrrid']

# 중복 여부 확인
print(f'train 중복 : {train.duplicated().sum()}')
print(f'val 중복 : {val.duplicated().sum()}')
print(f'test 중복 : {test.duplicated().sum()}')
print(f'unmatched 중복 : {unmatched.duplicated().sum()}')

# 서로 중복되는 nsrrid 확인
print(f'train & val 중복 : {train.isin(val).sum()}')
print(f'train & test 중복 : {train.isin(test).sum()}')
print(f'train & unmatched 중복 : {train.isin(unmatched).sum()}')
print(f'val & test 중복 : {val.isin(test).sum()}')
print(f'val & unmatched 중복 : {val.isin(unmatched).sum()}')
print(f'test & unmatched 중복 : {test.isin(unmatched).sum()}')

# unmatched를 test에 추가
test = pd.concat([test, unmatched])
print(f'test에 unmatched 추가 후 중복 : {test.duplicated().sum()}')

# 개수 확인
print(f'train: {train.shape}')
print(f'val: {val.shape}')
print(f'test: {test.shape}')

# 파일로 저장
train.to_csv(pwd + '/05_0_htnderv_s1_train_nsrrid.csv', index=False)
val.to_csv(pwd + '/05_0_htnderv_s1_val_nsrrid.csv', index=False)
test.to_csv(pwd + '/05_0_htnderv_s1_test_nsrrid.csv', index=False)
