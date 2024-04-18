from lightgbm import early_stopping
import pandas as pd
from preprocess import run
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPool1D
from keras.optimizers.legacy import Adam, RMSprop
from keras.callbacks import EarlyStopping

from keras.models import Model
import xgboost as xgb

from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

def CNN_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    model.add(Conv1D(256, 4, activation='relu', input_shape=(X_shape, 1), padding='same'))
    model.add(Dropout(dropout))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Dropout(dropout))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Conv1D(64, 2, activation='relu', padding='same'))
    model.add(Dropout(dropout))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(8, activation='softmax'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def DNN_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X_shape,)))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation='softmax'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def XGBoost_create(model=Sequential(), X_train=[], y_train=[], X_val=[], y_val=[]):
    without_output = Model(inputs=model.input, outputs=model.layers[-2].output)
    dtrain = xgb.DMatrix(without_output.predict(X_train), label=y_train)
    dval = xgb.DMatrix(without_output.predict(X_val), label=y_val)
    w_list = [(dtrain, 'train'), (dval, 'val')]
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'n_jobs': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }
    early_stopping_rounds = 100
    num_boost_round = 500
    # ml_model = xgb.XGBClassifier()
    ml_model=xgb.train(params, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds, evals=w_list, verbose_eval=100)
    # ml_model=xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=w_list, verbose_eval=100)
    return without_output, ml_model

if __name__=='__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240314'
    train = pd.read_csv(pwd+'/train.csv')
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    all_data = pd.read_csv(pwd+'/shhs1-dataset-0.20.0.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = []
    
    # 아래는 일단 나오긴 함
    # columns = ['nsaid1']
    # columns = ['nsaid1', 'thry1', 'lipid1', 'progst1', 'estrgn1', 'benzod1', 'ntca1', 'alpha1', 'ostrd1', 'sympth1', 'premar1', 'niac1', 'tca1', 'hctz1', 'hctzk1', 'istrd1', 'loop1', 'dig1', 'pdei1', 'ntg1', 'ccbir1', 'warf1', 'alphad1', 'anar1b1', 'anar1c1', 'anar31', 'basq1', 'betad1', 'anar1a1', 'aced1']
    # columns.extend(['hstg342p', 'timest1p', 'pctsa95h', 'remlaiip', 'timest2', 'times34p', 'waso', 'timest34', 'pcstahar', 'timest1', 'timerem', 'nremepop', 'stonsetp', 'hslptawp', 'pslp_hp3', 'remlaip'])
    # columns.extend(['ahi_c0h3a', 'rdinr3p', 'rdirem0p', 'rdi2pa', 'rdi0p', 'rdi0pns', 'rdinr0p', 'rdirop', 'rdirem2p', 'rdi5p', 'ahi_c0h3', 'rdinr2p', 'rdinop', 'rdinop3', 'rdiroa', 'rdi3pns', 'rdi0pa', 'ahi_c0h4', 'rdi2p', 'rdi2pns', 'rdi4pa'])
    # columns.extend(['hnroa3', 'hnroa5', 'hnrbp4', 'hnrop', 'hrop', 'hnrop2', 'hnrop3', 'hrop2', 'hnroa2', 'hnrbp', 'hnrbp2', 'hroa3', 'hnroa', 'hroa2', 'hrembp', 'hnrba', 'hroa4', 'hroa', 'hrop3', 'hnroa4', 'hroa5', 'hnrba3', 'hrop4', 'hnrop5', 'hnrba2', 'hrop5', 'oanop5', 'hnrbp5', 'hrembp5', 'hnrop4'])
    # 아래는 240402_DNN_XGB.py에서 사용한 컬럼
    columns = ['ccb1','ace1','beta1','diuret1','age_s1','dig1','waso','vaso1','aai','hctz1','chol','rcrdtime','mcs_s1','mi2slp02','stloutp','ntg1','timebedp','stonsetp','trig','timest2','twuweh02','avsao2nh','hremt2p','avdnop4','ahremop','slplatp','timest1p']
    # 아래는 240312에서 사용한 컬럼
    # columns.extend(['rdinr0p', 'rdi0ps', 'rdirem5p', 'rdirem4p', 'rdirem3p', 'rdi0p', 'rdirem2p', 'rdi0pa', 'rdi2pa', 'rdi3ps', 'rdinr2p', 'rdi2ps', 'rdi5p'])
    
    columns = list(set(columns))
    _, _, _, _, X_all, y_all = run(train, val, all_data, strategy, columns)
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns)

    from keras.models import load_model
    keras_model = load_model('CNN')
    xgb_model = xgb.Booster()
    xgb_model.load_model('XGBoost.json')
    bef_xgb = Model(inputs=keras_model.input, outputs=keras_model.layers[-2].output)

    cnn_y_pred = keras_model.predict(X_test)
    cnn_y_pred = [1 if x > 0.5 else 0 for x in cnn_y_pred]
    cnn_test_loss, cnn_test_acc = keras_model.evaluate(X_test, y_test)
    cnn_f1 = f1_score(y_test, cnn_y_pred)

    dtest = xgb.DMatrix(bef_xgb.predict(X_test), label=y_test)
    xgb_y_pred = xgb_model.predict(dtest)
    xgb_y_pred = [1 if x > 0.5 else 0 for x in xgb_y_pred]
    xgb_test_acc = (xgb_y_pred == y_test).mean()
    xgb_f1 = f1_score(y_test, xgb_y_pred)

    ##############################
    # 합쳐보기
    sum_pred = [x+y for x, y in zip(cnn_y_pred, xgb_y_pred)]
    sum_pred = [1 if x >1 else 0 for x in sum_pred]
    sum_acc = (sum_pred == y_test).mean()
    sum_f1 = f1_score(y_test, sum_pred)
    ##############################
    print(f'CNN test accuracy: {cnn_test_acc}, CNN f1 score: {cnn_f1}, CNN test loss: {cnn_test_loss}')
    print(f'XGB test accuracy: {xgb_test_acc}, XGB f1 score: {xgb_f1}')

    print(f'Sum accuracy: {sum_acc}, Sum f1 score: {sum_f1}')

    cnn_cm = confusion_matrix(y_test, cnn_y_pred)
    xgb_cm = confusion_matrix(y_test, xgb_y_pred)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cnn_cm, cmap='Blues')
    ax[0].set_title('CNN')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(['0', '1'])
    ax[0].set_yticklabels(['0', '1'])
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, cnn_cm[i, j], ha='center', va='center', color='black')
    ax[1].imshow(xgb_cm, cmap='Blues')
    ax[1].set_title('XGBoost')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')
    ax[1].set_xticks([0, 1])
    ax[1].set_yticks([0, 1])
    ax[1].set_xticklabels(['0', '1'])
    ax[1].set_yticklabels(['0', '1'])
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, xgb_cm[i, j], ha='center', va='center', color='black')
    plt.show()

    ##############################
    # test -> all_data

    cnn_y_pred = keras_model.predict(X_all)
    cnn_y_pred = [1 if x > 0.5 else 0 for x in cnn_y_pred]
    cnn_test_loss, cnn_test_acc = keras_model.evaluate(X_all, y_all)
    cnn_f1 = f1_score(y_all, cnn_y_pred)

    dall = xgb.DMatrix(bef_xgb.predict(X_all), label=y_all)
    xgb_y_pred = xgb_model.predict(dall)
    xgb_y_pred = [1 if x > 0.5 else 0 for x in xgb_y_pred]
    xgb_test_acc = (xgb_y_pred == y_all).mean()
    xgb_f1 = f1_score(y_all, xgb_y_pred)

    ##############################
    # 합쳐보기
    sum_pred = [x+y for x, y in zip(cnn_y_pred, xgb_y_pred)]
    sum_pred = [1 if x >1 else 0 for x in sum_pred]
    sum_acc = (sum_pred == y_all).mean()
    sum_f1 = f1_score(y_all, sum_pred)
    ##############################
    print(f'CNN test accuracy: {cnn_test_acc}, CNN f1 score: {cnn_f1}, CNN test loss: {cnn_test_loss}')
    print(f'XGB test accuracy: {xgb_test_acc}, XGB f1 score: {xgb_f1}')

    print(f'Sum accuracy: {sum_acc}, Sum f1 score: {sum_f1}')

    ##############################
    import tensorflow as tf
    import keras.losses
    import numpy as np
    import seaborn as sns
    def extract_feature_importance(model, X, y, columns):
        input_tensor = tf.convert_to_tensor(X)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            prediction = model(input_tensor)
            y = np.expand_dims(y, axis=1)
            loss = keras.losses.binary_crossentropy(y, prediction)
            gradient = tape.gradient(loss, input_tensor)

        feature_importance = np.mean(np.abs(gradient.numpy()), axis=0)
        
        feature_match = list(zip(columns, feature_importance))
        feature_match = sorted(feature_match, key=lambda x: x[1], reverse=True)
        print(feature_match[:10])

        # feature_match plot
        # feature_match의 0번째 인덱스가 변수명, 1번째 인덱스가 중요도
        feature_name = [x[0] for x in feature_match[:30]]
        importance = [x[1] for x in feature_match[:30]]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=importance, y=feature_name, hue=feature_name, legend=False, palette="viridis")
        plt.title("Feature Importance")
        plt.xlabel("Feature Importance")
        plt.ylabel("Variable Name")
        plt.show()

    if columns == []:
        columns = list(train.columns)
    extract_feature_importance(keras_model, X_train, y_train, columns)
    # extract_feature_importance(keras_model, X_all, y_all, columns)
