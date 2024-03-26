import pandas as pd
import numpy as np
class preprocess:
    def __init__(self, train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame):
        self.train = train
        self.val = val
        self.test = test
    
    def column_replace(self, strategy=pd.DataFrame):
        # val, test의 컬럼명을 strategy에서 찾아 결측치를 행 데이터의 값으로 대체, NaN은 컬럼 제거
        for col in strategy.columns.tolist():
            fill_value = strategy[[col]].values[0][0]
            if fill_value == 9999999999:
                self.val = self.val.drop(col, axis=1)
                self.test = self.test.drop(col, axis=1)
            else:
                self.val[col] = self.val[col].fillna(fill_value)
                self.test[col] = self.test[col].fillna(fill_value)

        new_val = self.val.copy()
        new_test = self.test.copy()

        self.val = self.date_to_int(new_val)
        self.test = self.date_to_int(new_test)
    
    def date_to_int(self, df):
        ## 시간 데이터 정수로 변환
        column_name = 'rcrdtime'
        change_name = column_name + "_numeric"
        df[column_name] = pd.to_datetime(df[column_name], format='%H:%M:%S', errors='coerce').dt.time
        df[change_name] = [(t.hour*3600 + t.minute*60 + t.second) for t in df[column_name]]
        df[column_name] = df[change_name]
        df = df.drop(columns=change_name)
        df = df.astype(np.float64)
        return df

    def return_data(self):
        return self.train, self.val, self.test

    def scaler(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(self.train)
        self.train = sc.transform(self.train)
        self.val = sc.transform(self.val)
        self.test = sc.transform(self.test)

def run(train, val, test, strategy, columns=list()):
    pre = preprocess(train, val, test)
    pre.column_replace(strategy)
    train, val, test = pre.return_data()

    y_train = train['htnderv_s1']
    X_train = train.drop('htnderv_s1', axis=1)
    y_val = val['htnderv_s1']
    X_val = val.drop('htnderv_s1', axis=1)
    y_test = test['htnderv_s1']
    X_test = test.drop('htnderv_s1', axis=1)

    if columns!=[]:
        X_train = X_train[columns]
        X_val = X_val[columns]
        X_test = X_test[columns]

    pre = preprocess(X_train, X_val, X_test)
    pre.scaler()
    X_train, X_val, X_test = pre.return_data()
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    print('직접 실행 X')
    print('import preprocess 혹은 from preprocess import run 사용')