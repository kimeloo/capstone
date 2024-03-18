import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 불러오기
pwd = '~/documents/coding/capstone/data/240201/'
data = pd.read_csv(pwd + '01_htnderv_s1.csv')

# train-test set split
X = data
X_train, X_test = train_test_split(X, test_size=0.3, random_state=2024)

# 각각 저장
X_train.to_csv(pwd + '02_htnderv_s1_train.csv', index=False)
X_test.to_csv(pwd + '02_htnderv_s1_test.csv', index=False)