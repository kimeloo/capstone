from sklearn.preprocessing import StandardScaler
import pandas as pd

def standard_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data

if __name__ == '__main__':
    pwd = '~/Documents/Coding/capstone/data/240314'
    data = pd.read_csv('data.csv')
    data = standard_scaler(data)
    pd.DataFrame(data).to_csv('data.csv', index=False)