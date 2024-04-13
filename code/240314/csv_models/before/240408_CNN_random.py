import pandas as pd
from preprocess import run
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.optimizers.legacy import Adam, RMSprop
from keras.callbacks import EarlyStopping

from keras.models import Model
import xgboost as xgb

from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

def CNN_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    model.add(Conv1D(256, 2, activation='relu', input_shape=(X_shape, 1)))
    model.add(Dropout(dropout))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation='softmax'))
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

def XGBoost_create(model=Sequential(), X_train=[], y_train=[]):
    without_output = Model(inputs=model.input, outputs=model.layers[-2].output)
    ml_model = xgb.XGBClassifier()
    ml_model.fit(without_output.predict(X_train), y_train)
    return without_output, ml_model

if __name__=='__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240314'
    train = pd.read_csv(pwd+'/train.csv')
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = ['hnrop', 'hrop', 'hnrop2', 'hrop2', 'hnroa2', 'hnroa3', 'hnrop3', 'hnrbp', 'hroa3', 'hnrbp2']
    
    # 아래는 일단 나오긴 함
    columns = ['diuret1', 'asa1', 'nsaid1', 'lipid1', 'alpha1', 'nitro15', 'premar1', 'thry1', 'progst1', 'ace1']
    columns = ['estrgn1', 'progst1', 'anar1a1', 'lipid1', 'sympth1', 'tca1', 'asa1', 'nsaid1', 'benzod1', 'premar1', 'pdei1', 'ntca1', 'warf1', 'loop1', 'hctz1', 'hctzk1', 'ccbir1', 'ccbsr1', 'alpha1', 'alphad1', 'anar1b1', 'anar1c1', 'anar31', 'basq1', 'niac1', 'thry1', 'istrd1', 'ostrd1', 'beta1', 'betad1', 'ccb1', 'ace1', 'aced1']
    columns = ['nsaid1', 'thry1', 'lipid1', 'progst1', 'estrgn1', 'benzod1', 'ntca1', 'alpha1', 'ostrd1', 'sympth1', 'premar1', 'niac1', 'tca1', 'hctz1', 'hctzk1', 'istrd1', 'loop1', 'dig1', 'pdei1', 'ntg1', 'ccbir1', 'warf1', 'alphad1', 'anar1b1', 'anar1c1', 'anar31', 'basq1', 'betad1', 'anar1a1', 'aced1']
    import random
    columns_list = []
    for _ in range(10):
        columns_list.append(random.sample(columns, 20))
    count = 0
    result = dict()
    for columns in columns_list:
        # 아래는 240402_DNN_XGB.py에서 사용한 컬럼
        # columns = ['ccb1','ace1','beta1','diuret1','age_s1','dig1','waso','vaso1','aai','hctz1','chol','rcrdtime','mcs_s1','mi2slp02','stloutp','ntg1','timebedp','stonsetp','trig','timest2','twuweh02','avsao2nh','hremt2p','avdnop4','ahremop','slplatp','timest1p']
        # 아래는 옛날거
        # columns = ['rdinr0p', 'rdi0ps', 'rdirem5p', 'rdirem4p', 'rdirem3p', 'rdi0p', 'rdirem2p', 'rdi0pa', 'rdi2pa', 'rdi3ps', 'rdinr2p', 'rdi2ps', 'rdi5p']
        X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns)

        # epochs_list = [100, 300, 500, 1000, 1500, 2000, 3000, 5000, 10000]
        epochs_list = [100]
        # optimizer_list = [Adam(learning_rate=0.0001), RMSprop(learning_rate=0.0001)]
        optimizer_list = [Adam(learning_rate=0.0001), Adam(learning_rate=0.001)]
        loss_list = ['binary_crossentropy']
        metrics_list = ['accuracy']
        # batch_size_list = [128, 256, 512]
        batch_size_list = [256]
        dropout_list = [0.5]
        X_shape = X_train.shape[1]

        # 모든 경우 수행
        for epochs in epochs_list:
            for optimizer in optimizer_list:
                for loss in loss_list:
                    for metrics in metrics_list:
                        for batch_size in batch_size_list:
                            for dropout in dropout_list:
                                keras_model = CNN_create(opt=optimizer, loss=loss, metrics=metrics, dropout=dropout, X_shape=X_shape)
                                keras_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
                                noout, xgb_model = XGBoost_create(model=keras_model, X_train=X_train, y_train=y_train)

                                cnn_y_pred = keras_model.predict(X_test)
                                cnn_y_pred = [1 if x > 0.5 else 0 for x in cnn_y_pred]
                                cnn_test_loss, cnn_test_acc = keras_model.evaluate(X_test, y_test)
                                cnn_f1 = f1_score(y_test, cnn_y_pred)

                                xgb_y_pred = xgb_model.predict(noout.predict(X_test))
                                xgb_y_pred = [1 if x > 0.5 else 0 for x in xgb_y_pred]
                                xgb_test_acc = xgb_model.score(noout.predict(X_test), y_test)
                                xgb_f1 = f1_score(y_test, xgb_y_pred)

                                print(f'epochs: {epochs}, optimizer: {optimizer}, loss: {loss}, metrics: {metrics}, batch_size: {batch_size}, dropout: {dropout}\ncnn_test_acc: {cnn_test_acc}, cnn_f1: {cnn_f1}, xgb_test_acc: {xgb_test_acc}, xgb_f1: {xgb_f1}')
                                result[(count, epochs, optimizer, loss, metrics, batch_size, dropout)] = [cnn_test_acc, cnn_f1, xgb_test_acc, xgb_f1]
                                count += 1
                                print(columns)
                                input()
    max_cnn_acc = 0
    max_cnn_param = None
    max_xgb_acc = 0
    max_xgb_param = None
    for r in result:
        if result[r][0] > max_cnn_acc:
            max_cnn_acc = result[r][0]
            max_cnn_param = r
        if result[r][2] > max_xgb_acc:
            max_xgb_acc = result[r][2]
            max_xgb_param = r
    print(f'Best cnn accuracy: {max_cnn_acc} with parameters {max_cnn_param}')
    print(f'Best xgb accuracy: {max_xgb_acc} with parameters {max_xgb_param}')
    print(f'Best cnn columns: {columns_list[max_cnn_param[0]]}')
    print(f'Best xgb columns: {columns_list[max_xgb_param[0]]}')