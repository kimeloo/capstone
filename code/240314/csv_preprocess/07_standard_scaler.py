from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd

class standard_scaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def standard_scaler(self, data):
        scaler = self.scaler
        scaler.fit(data)
        data = scaler.transform(data)
        return data
    
    def dump(self, pwd):
        dump(self.scaler, pwd+'/standard_scaler.bin', compress=True)

if __name__ == '__main__':
    pwd = '~/Documents/Coding/capstone/data/240314'
    data = pd.read_csv(pwd+'/04_htnderv_s1_train.csv')
    X_train = data.drop(columns=['htnderv_s1'])
    sc = standard_scaler()
    X_train = sc.standard_scaler(X_train)
    sc.dump("/users/kimeloo"+pwd[1:])