import pandas as pd
from preprocess import run
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

class accuracy():
    def __init__(self, history):
        self.history = history

    def run(self, X_test, y_test, y_pred=[]):
        self.X_test = X_test
        self.y_test = y_test
        if len(y_pred)==0:
            self.y_pred = (self.history.model.predict(X_test) > 0.5).astype("int32")
        self._eval()
        self._f1()
        self._plot(1, 3, False)
        self._plot_confusion_matrix(3, 3, False)
        plt.show()

    def _eval(self):
        test_loss, test_acc = self.history.model.evaluate(self.X_test, self.y_test)
        print(f'Test loss: {test_loss}\nTest accuracy: {test_acc}')

    def _f1(self):
        f1 = f1_score(self.y_test, self.y_pred)
        print("F1 Score:", f1)
    
    def _plot(self, location=1, len=2, plot=True):
        # 정확도 그래프
        plt.subplot(1, len, location)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        # 오차 그래프
        plt.subplot(1, len, location+1)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.legend(['Training', 'Validation'])
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if plot:
            plt.show()

    def _plot_confusion_matrix(self, location, len, plot=True):
        conf_mat = confusion_matrix(self.y_test, self.y_pred)
        plt.subplot(1, len, location)
        plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(conf_mat[i, j]), ha='center', va='center', fontsize=20, color='red')
        if plot:
            plt.show()

from keras.models import Model
import xgboost as xgb
class mymodel():
    def __init__(self, num_features, dropout=0.0):
        self.model = Sequential()
        self.num_features = num_features
        self.dropout = dropout
    
    def _DNN(self):
        self.model.add(Dense(256, activation='relu', kernel_regularizer=l1(0.0001), input_shape=(self.num_features,)))
        self.model.add(Dropout(self.dropout)) if self.dropout>0 else None
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(self.dropout)) if self.dropout>0 else None
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(self.dropout)) if self.dropout>0 else None
        self.model.add(Dense(8, activation='tanh'))
        self.model.add(Dense(1, activation='sigmoid'))

    def _XGBoost(self):
        intermediate_model = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
        mlmodel = xgb.XGBClassifier()
        mlmodel.fit(intermediate_model.predict(X_train), y_train)
        return intermediate_model, mlmodel

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def build(self):
        self._DNN()

if __name__ == '__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240314'
    train = pd.read_csv(pwd+'/train.csv')
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = ['ccb1','ace1','beta1','diuret1','age_s1','dig1','waso','vaso1','aai','hctz1','chol','rcrdtime','mcs_s1','mi2slp02','stloutp','ntg1','timebedp','stonsetp','trig','timest2','twuweh02','avsao2nh','hremt2p','avdnop4','ahremop','slplatp','timest1p']
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns=[])

    # 모델 설계
    num_features = X_train.shape[1]
    model = mymodel(num_features, 0.3)
    model.build()
    model.compile(optimizer=Adam(learning_rate=0.0001))

    # 모델 학습
    epoch = 1000
    batch_size = 512
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # history = model.model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
    history = model.model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_val, y_val))

    # DNN 모델 평가
    acc = accuracy(history)
    acc.run(X_test, y_test)

    # XGBoost 모델
    dnn, xgboost = model._XGBoost()
    y_pred = xgboost.predict(dnn(X_test))

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    import numpy as np
    # hybrid 모델 평가
    accu = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accu}\nF1 Score: {f1}')
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf_mat[i, j]), ha='center', va='center', color='red')
    plt.xticks(np.arange(2), ['No HTN', 'HTN'])
    plt.yticks(np.arange(2), ['No HTN', 'HTN'])
    plt.show()

    print(model.model.summary())
    print(dnn.summary())