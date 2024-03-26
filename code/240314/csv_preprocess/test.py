import pandas as pd

pwd = '~/Documents/Coding/capstone/data/240314'
original_filename = 'shhs1-dataset-0.20.0.csv'

df = pd.read_csv(pwd + '/' + original_filename, low_memory=False)

# htnderv_s1 개수 확인
print(df['htnderv_s1'].value_counts())
print(df.shape)