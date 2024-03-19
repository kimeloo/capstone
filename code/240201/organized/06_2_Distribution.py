import pandas as pd
import matplotlib.pyplot as plt

pwd = '~/Documents/Coding/capstone/data/240201'
before = pd.read_csv(pwd+'/03_htnderv_s1_train.csv')
after_psm = pd.read_csv(pwd+'/04_htnderv_s1_train.csv')

column_names = ['height', 'weight', 'weight20', 'age_s1', 'gender', 'race']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

axes = axes.flatten()

for i, column in enumerate(column_names):
    axes[i].hist(before[column], alpha=0.5, label='before', color='blue', bins=20)
    axes[i].hist(after_psm[column], alpha=0.5, label='after_psm', color='red', bins=20)
    axes[i].set_title(f'Distribution of {column} in before and after_psm')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.tight_layout()
plt.show()