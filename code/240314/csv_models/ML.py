import pandas as pd
from preprocess import run

class ml_model:
    def __init__(self, train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame, columns=list()):
        self.train = train
        self.val = val
        self.test = test
        self.columns = columns
    
    def select(self, ):
        pass

if __name__ == '__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240314'
    train = pd.read_csv(pwd+'/train.csv')
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    # columns = ['rdirem4p', 'rdirem5p', 'rdinr0p', 'rdinr2p', 'rdinr3p', 'rdinr4p', 'rdinr5p']
    columns = ['ccb1','ace1','beta1','diuret1','age_s1','dig1','waso','vaso1','aai','hctz1']
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns)

    # 위에서 파일명, 경로 수정하고 아래부터 코드 작성

    # randomforest 모델
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    # xgboost 모델
    from xgboost import XGBClassifier

    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    # lightgbm 모델
    from lightgbm import LGBMClassifier

    lgbm = LGBMClassifier(n_estimators=100, random_state=42)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    print(accuracy_score(y_test, y_pred))
