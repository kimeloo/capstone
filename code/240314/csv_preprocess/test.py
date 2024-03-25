import pandas as pd

pwd = '~/Documents/Coding/capstone/data/240314'
filename = 'strategy.csv'
df = pd.read_csv(f'{pwd}/{filename}')

# nan 개수 출력
print(df.isnull().sum().sum())