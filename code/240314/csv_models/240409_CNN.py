import pandas as pd
from preprocess import run
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPool1D
from keras.optimizers.legacy import Adam, RMSprop

from keras.models import Model
import xgboost as xgb

from sklearn.metrics import f1_score

def CNN_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    model.add(Conv1D(256, 4, activation='relu', input_shape=(X_shape, 1)))
    model.add(Dropout(dropout))
    model.add(MaxPool1D(pool_size=4, strides=2))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPool1D(pool_size=3, strides=2))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Flatten())
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
    ml_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', max_depth=6, learing_rate=0.2, n_estimators=100, n_jobs=-1, subsample=0.8, colsample_bytree=0.8, random_state=42)
    ml_model.fit(without_output.predict(X_train), y_train)
    return without_output, ml_model

if __name__=='__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240314'
    train = pd.read_csv(pwd+'/train.csv')
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = []
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns)

    epochs_list = [500, 1000, 3000]
    optimizer_list = [Adam(learning_rate=0.001), Adam(learning_rate=0.0001), RMSprop(learning_rate=0.001), RMSprop(learning_rate=0.0001)]
    loss_list = ['binary_crossentropy']
    metrics_list = ['accuracy']
    batch_size_list = [256]
    dropout_list = [0.2, 0.3, 0.5]
    X_shape = X_train.shape[1]
    params = [epochs_list, optimizer_list, loss_list, metrics_list, batch_size_list, dropout_list]

    # 모든 경우 수행
    params_list = []
    for epochs in epochs_list:
        for optimizer in optimizer_list:
            for loss in loss_list:
                for metrics in metrics_list:
                    for batch_size in batch_size_list:
                        for dropout in dropout_list:
                            params_list.append((epochs, optimizer, loss, metrics, batch_size, dropout))
    
    result = dict()
    while params_list:
        epochs, optimizer, loss, metrics, batch_size, dropout = params_list.pop()
        keras_model = CNN_create(opt=optimizer, loss=loss, metrics=metrics, dropout=dropout, X_shape=X_shape)
        keras_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
        bef_xgb, xgb_model = XGBoost_create(model=keras_model, X_train=X_train, y_train=y_train)

        cnn_y_pred = keras_model.predict(X_test)
        cnn_y_pred = [1 if x > 0.5 else 0 for x in cnn_y_pred]
        cnn_test_loss, cnn_test_acc = keras_model.evaluate(X_test, y_test)
        cnn_f1 = f1_score(y_test, cnn_y_pred)

        xgb_y_pred = xgb_model.predict(bef_xgb.predict(X_test))
        xgb_y_pred = [1 if x > 0.5 else 0 for x in xgb_y_pred]
        xgb_test_acc = xgb_model.score(bef_xgb.predict(X_test), y_test)
        xgb_f1 = f1_score(y_test, xgb_y_pred)

        print(f'epochs: {epochs}, optimizer: {optimizer.get_config()["name"]}({optimizer.get_config()["learning_rate"]}), loss: {loss}, metrics: {metrics}, batch_size: {batch_size}, dropout: {dropout}')
        result[(epochs, optimizer, loss, metrics, batch_size, dropout)] = [cnn_test_acc, cnn_f1, xgb_test_acc, xgb_f1]
    max_cnn_acc = 0
    max_cnn_param = None
    max_xgb_acc = 0
    max_xgb_param = None
    for r in result:
        if result[r][0] > max_cnn_acc:
            max_cnn_acc = result[r][0]
            temp = list(r)
            temp[1] = (temp[1].get_config()['name'], temp[1].get_config()['learning_rate'])
            max_cnn_param = temp
        if result[r][2] > max_xgb_acc:
            max_xgb_acc = result[r][2]
            temp = list(r)
            temp[1] = (temp[1].get_config()['name'], temp[1].get_config()['learning_rate'])
            max_xgb_param = temp
    print(f'Best cnn accuracy: {max_cnn_acc} with parameters {max_cnn_param}')
    print(f'Best xgb accuracy: {max_xgb_acc} with parameters {max_xgb_param}')