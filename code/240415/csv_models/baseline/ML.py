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
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240415'
    train = pd.read_csv(pwd+'/train_replaced.csv')
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = []
    columns = ['ccb1', 'ace1', 'beta1', 'diuret1', 'age_s1', 'dig1', 'vaso1', 'waso', 'aai', 'hctzk1']
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns)

    # 위에서 파일명, 경로 수정하고 아래부터 코드 작성

    # # randomforest 모델
    # from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)
    # print(accuracy_score(y_test, y_pred))

    # xgboost 모델
    from xgboost import XGBClassifier
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    xgb = XGBClassifier(n_estimators=5000, random_state=42, n_jobs=-1, eta=0.2)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    xgb_cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(xgb_cm, cmap='Blues')
    ax[0].set_title('CNN')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(['0', '1'])
    ax[0].set_yticklabels(['0', '1'])
    for i in range(2):
        for j in range(2):
            if i+j==0 or i*j==1:
                ax[0].text(j, i, xgb_cm[i, j], ha='center', va='center', color='white')
            else:
                ax[0].text(j, i, xgb_cm[i, j], ha='center', va='center', color='black')
    plt.show()

    # # lightgbm 모델
    # from lightgbm import LGBMClassifier

    # lgbm = LGBMClassifier(n_estimators=100, random_state=42)
    # lgbm.fit(X_train, y_train)
    # y_pred = lgbm.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
