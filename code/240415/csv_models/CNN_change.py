import pandas as pd
from preprocess import run
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPool1D
from keras.optimizers.legacy import Adam, RMSprop
from keras.callbacks import EarlyStopping

from keras.models import Model
import xgboost as xgb

from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt

def CNN_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    # model.add(Conv1D(128, 2, activation='relu', input_shape=(X_shape, 1), padding='same'))
    # model.add(Conv1D(64, 2, activation='relu', padding='same'))
    model.add(Conv1D(32, 2, activation='relu', input_shape=(X_shape, 1), padding='same'))
    model.add(Dropout(dropout))
    model.add(Conv1D(16, 2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Conv1D(8, 2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def DNN_create(opt='adam', loss='binary_crossentropy', metrics=['accuracy'], dropout=0.3, X_shape=1000):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_shape,)))
    model.add(Dense(16, activation='softmax'))
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
        'eta': 0.01,
        'max_depth': 15,
        'n_jobs': -1,
        # 'subsample': 0.9,
        # 'lambda': 0.1,
        # 'alpha': 0.1,
        'random_state': 42,
    }
    early_stopping_rounds = 100
    num_boost_round = 500
    # ml_model = xgb.XGBClassifier()
    ml_model=xgb.train(params, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds, evals=w_list, verbose_eval=100)
    # ml_model=xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=w_list, verbose_eval=100)
    return without_output, ml_model

if __name__=='__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240415'
    train = pd.read_csv(pwd+'/train_replaced.csv', low_memory=False)
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    all_data = pd.read_csv(pwd+'/shhs1-dataset-0.20.0.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = []
    
    # import dict_list as dl
    # columns = ['ccb1', 'ace1', 'beta1', 'diuret1', 'dig1', 'vaso1', 'chol', 'hctzk1', 'hctz1', 'timebedp', 'trig', 'stonsetp', 'ntg1', 'age_s1', 'waso', 'aai', 'height', 'stloutp', 'pcs_s1', 'timest1p', 'mi2slp02', 'slpeffp', 'ahremop', 'nrvous25', 'loop1', 'twuweh02', 'rcrdtime', 'mcs_s1', 'remlaip', 'alcoh']
    # columns = ['timebedp', 'trig', 'stonsetp', 'age_s1', 'waso', 'aai', 'height', 'stloutp', 'pcs_s1', 'timest1p', 'mi2slp02', 'slpeffp', 'ahremop', 'nrvous25', 'twuweh02', 'rcrdtime', 'mcs_s1', 'remlaip', 'alcoh']

    columns = ['ccb1', 'ace1', 'beta1', 'diuret1', 'age_s1', 'dig1', 'vaso1', 'waso', 'aai', 'hctzk1']
    columns = list(set(columns))
    _, _, _, _, X_all, y_all = run(train, val, all_data, strategy, columns)
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns)

    epochs = 500
    optimizer = Adam(learning_rate=0.0001)
    loss = 'binary_crossentropy'
    metrics = 'accuracy'
    batch_size = 512
    dropout = 0.1
    X_shape = X_train.shape[1]
    temp = CNN_create(X_shape=X_shape)
    print(temp.input_shape)
    print(temp.summary())
    input()

    keras_model = CNN_create(opt=optimizer, loss=loss, metrics=metrics, dropout=dropout, X_shape=X_shape)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # keras_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1, callbacks=[early_stopping])
    keras_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)

    keras_model.save('CNN')
    # from keras.models import load_model
    # keras_model = load_model('CNN')
    bef_xgb, xgb_model = XGBoost_create(model=keras_model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    cnn_y_pred = keras_model.predict(X_test)
    cnn_y_pred = [1 if x > 0.5 else 0 for x in cnn_y_pred]
    cnn_test_loss, cnn_test_acc = keras_model.evaluate(X_test, y_test)
    cnn_f1 = f1_score(y_test, cnn_y_pred)
    cnn_recall = recall_score(y_test, cnn_y_pred)
    cnn_precision = precision_score(y_test, cnn_y_pred)

    dtest = xgb.DMatrix(bef_xgb.predict(X_test), label=y_test)
    xgb_y_pred = xgb_model.predict(dtest)
    xgb_y_pred = [1 if x > 0.5 else 0 for x in xgb_y_pred]
    xgb_test_acc = (xgb_y_pred == y_test).mean()
    xgb_f1 = f1_score(y_test, xgb_y_pred)
    xgb_recall = recall_score(y_test, xgb_y_pred)
    xgb_precision = precision_score(y_test, xgb_y_pred)

    ##############################
    # 합쳐보기
    sum_pred = [x+y for x, y in zip(cnn_y_pred, xgb_y_pred)]
    sum_pred = [1 if x >1 else 0 for x in sum_pred]
    sum_acc = (sum_pred == y_test).mean()
    sum_f1 = f1_score(y_test, sum_pred)
    ##############################
    # print(f'epochs: {epochs}, optimizer: {optimizer.get_config()["name"]}({optimizer.get_config()["learning_rate"]}), loss: {loss}, metrics: {metrics}, batch_size: {batch_size}, dropout: {dropout}')
    print(f'CNN test accuracy: {cnn_test_acc}, CNN recall: {cnn_recall}, CNN precision: {cnn_precision}, CNN f1 score: {cnn_f1}, CNN test loss: {cnn_test_loss}')
    print(f'XGB test accuracy: {xgb_test_acc}, XGB recall: {xgb_recall}, XGB precision: {xgb_precision}, XGB f1 score: {xgb_f1}')

    print(f'Sum accuracy: {sum_acc}, Sum f1 score: {sum_f1}')



    xgb_model.save_model('XGBoost.json')
    print('Models saved')

    print(keras_model.summary())

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
            if i+j==0 or i*j==1:
                ax[0].text(j, i, cnn_cm[i, j], ha='center', va='center', color='white')
            else:
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
            if i+j==0 or i*j==1:
                ax[1].text(j, i, xgb_cm[i, j], ha='center', va='center', color='white')
            else:
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
    # print(f'epochs: {epochs}, optimizer: {optimizer.get_config()["name"]}({optimizer.get_config()["learning_rate"]}), loss: {loss}, metrics: {metrics}, batch_size: {batch_size}, dropout: {dropout}')
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
        print(feature_match)

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
    # extract_feature_importance(keras_model, X_train, y_train, columns)
    # extract_feature_importance(keras_model, X_all, y_all, columns)
