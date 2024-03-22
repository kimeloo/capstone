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

# 파일별 nsrrid, htnderv_s1 추출
train = train_csv[['nsrrid', 'htnderv_s1']]
val = val_csv[['nsrrid', 'htnderv_s1']]
test = test_csv[['nsrrid', 'htnderv_s1']]
unmatched = unmatched_csv[['nsrrid', 'htnderv_s1']]

# 중복 여부 확인
print(f'train 중복 : {train.duplicated().sum()}')
print(f'val 중복 : {val.duplicated().sum()}')
print(f'test 중복 : {test.duplicated().sum()}')
print(f'unmatched 중복 : {unmatched.duplicated().sum()}')

# unmatched를 test에 추가
test = pd.concat([test, unmatched])
print(f'test에 unmatched 추가 후 중복 : {test.duplicated().sum()}')

# stratified random sampling
def stratified_random_sampling(df, column_name):
    counts = df[column_name].value_counts()
    sample_size = min(counts)
    columns_to_keep = ['nsrrid', column_name]
    stratified_sample = df.groupby(column_name)[columns_to_keep].apply(lambda x: x.sample(sample_size, random_state=2024))
    stratified_sample.reset_index(drop=True, inplace=True)
    return stratified_sample

val = stratified_random_sampling(val, 'htnderv_s1')
test = stratified_random_sampling(test, 'htnderv_s1')

# 개수 확인
print('val, test 1:1 샘플링 완료')
print(f"train: {train['htnderv_s1'].value_counts()}")
print(f"val: {val['htnderv_s1'].value_counts()}")
print(f"test: {test['htnderv_s1'].value_counts()}")

# 파일로 저장
train.to_csv(pwd + '/train_nsrrid_htnderv_s1.csv', index=False)
val.to_csv(pwd + '/val_nsrrid_htnderv_s1.csv', index=False)
test.to_csv(pwd + '/test_nsrrid_htnderv_s1.csv', index=False)
