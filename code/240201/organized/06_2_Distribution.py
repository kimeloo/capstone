import pandas as pd
import matplotlib.pyplot as plt

pwd = '~/Documents/Coding/capstone/data/240201'
before = pd.read_csv(pwd+'/03_htnderv_s1_train.csv')
after_psm = pd.read_csv(pwd+'/04_htnderv_s1_train.csv')

before_0 = before[before['htnderv_s1'] == 0]
after_0 = after_psm[after_psm['htnderv_s1'] == 0]
before_1 = before[before['htnderv_s1'] == 1]
after_1 = after_psm[after_psm['htnderv_s1'] == 1]

column_names = ['bmi_s1', 'age_s1', 'gender']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

axes = axes.flatten()

for i, column in enumerate(column_names):
    axes[i].hist(after_0[column], alpha=0.5, label='normal', color='blue', bins=20)
    axes[i].hist(after_1[column], alpha=0.5, label='HTN', color='red', bins=20)
    axes[i].set_title(f'Distribution of {column} in after PS matching')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.tight_layout()
plt.show()
