import pandas as pd
from preprocess import run
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPool1D, LSTM
from keras.optimizers.legacy import Adam, RMSprop
from keras.callbacks import EarlyStopping

from keras.models import Model
import xgboost as xgb

from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt

def CNN_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape=(X_shape, 1))) 
    model.add(MaxPool1D(2))
    model.add(Conv1D(16, 2, activation='relu')) 
    model.add(Flatten())
    model.add(Dense(8, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def DNN_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_shape,)))
    model.add(Dense(16, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def LSTM_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_shape, 1)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def XGBoost_baseline(X_train=[], y_train=[], X_val=[], y_val=[]):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    w_list = [(dtrain, 'train'), (dval, 'val')]
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
    }
    early_stopping_rounds = 100
    num_boost_round = 500
    # ml_model = xgb.XGBClassifier()
    ml_model=xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=w_list, verbose_eval=100)
    return ml_model

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
    ml_model=xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=w_list, verbose_eval=100)
    return without_output, ml_model

if __name__=='__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240415'
    train = pd.read_csv(pwd+'/train_replaced.csv', low_memory=False)
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    # all_data = pd.read_csv(pwd+'/shhs1-dataset-0.20.0.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = []
    columns = ['ccb1', 'ace1', 'beta1', 'diuret1', 'age_s1', 'dig1', 'vaso1', 'waso', 'aai', 'hctzk1']
    columns = list(set(columns))
    # _, _, _, _, X_all, y_all = run(train, val, all_data, strategy, columns)
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns)

    model_type = 'CNN'

    epochs = 100
    optimizer = Adam(learning_rate=0.0001)
    loss = 'binary_crossentropy'
    metrics = 'accuracy'
    batch_size = 512
    dropout = 0.2
    X_shape = X_train.shape[1]

    keras_model = CNN_create(opt=optimizer, loss=loss, metrics=metrics, dropout=dropout, X_shape=X_shape)
    keras_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)

    cnn_y_pred = keras_model.predict(X_test)
    cnn_y_pred = [1 if x > 0.5 else 0 for x in cnn_y_pred]
    cnn_test_loss, cnn_test_acc = keras_model.evaluate(X_test, y_test)
    cnn_recall = recall_score(y_test, cnn_y_pred)
    cnn_precision = precision_score(y_test, cnn_y_pred)
    cnn_f1 = f1_score(y_test, cnn_y_pred)

    print(f'epochs: {epochs}, optimizer: {optimizer.get_config()["name"]}({optimizer.get_config()["learning_rate"]}), loss: {loss}, metrics: {metrics}, batch_size: {batch_size}, dropout: {dropout}')
    print(f'test accuracy: {cnn_test_acc}, recall: {cnn_recall}, precision: {cnn_precision}, f1 score: {cnn_f1}, test loss: {cnn_test_loss}')

    cnn_cm = confusion_matrix(y_test, cnn_y_pred)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cnn_cm, cmap='Blues')
    ax[0].set_title(model_type)
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(['0', '1'])
    ax[0].set_yticklabels(['0', '1'])
    for i in range(2):
        for j in range(2):
            if i+j==0 or i*j==1:
                ax[0].text(j, i, cnn_cm[i, j], ha='center', va='center', color='white')
            else:
                ax[0].text(j, i, cnn_cm[i, j], ha='center', va='center', color='black')
    plt.show()