import pandas as pd
from preprocess import run
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1

from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

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
        self.test()
        self.f1()
        self.plot(1, 3, False)
        self.plot_confusion_matrix(3, 3, False)
        plt.show()

    def test(self):
        test_loss, test_acc = self.history.model.evaluate(self.X_test, self.y_test)
        print(f'Test loss: {test_loss}\nTest accuracy: {test_acc}')

    def f1(self):
        f1 = f1_score(self.y_test, self.y_pred)
        print("F1 Score:", f1)
    
    def plot(self, location=1, len=2, plot=True):
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

    def plot_confusion_matrix(self, location, len, plot=True):
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
    
    def _default_1DCNN(self):
        self.model.add(Conv1D(256, kernel_size=2, activation='relu', input_shape=(self.num_features, 1)))
        self.model.add(Conv1D(128, kernel_size=2, activation='relu'))
        self.model.add(Dropout(self.dropout)) if self.dropout>0 else None
    
    def _default_DNN(self):
        self.model.add(Dense(256, activation='relu', kernel_regularizer=l1(0.0001), input_shape=(self.num_features,)))
        self.model.add(Dropout(self.dropout)) if self.dropout>0 else None
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(self.dropout)) if self.dropout>0 else None
    
    def _hidden(self):
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(self.dropout)) if self.dropout>0 else None
        self.model.add(Dense(8, activation='softmax'))
        # self.model.add(Dropout(self.dropout)) if self.dropout>0 else None

    def _output(self, model):
        if model=='DNN':
            self.model.add(Dense(1, activation='sigmoid'))
        elif model=='1DCNN':
            self.model.add(Flatten())

    def _XGBoost(self):
        intermediate_model = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
        mlmodel = xgb.XGBClassifier()
        mlmodel.fit(intermediate_model.predict(X_train), y_train)
        return intermediate_model, mlmodel

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        return self.model

    def build(self, model='DNN'):
        if model == 'DNN':
            self._default_DNN()
        elif model == '1DCNN':
            self._default_1DCNN()
        self._hidden()
        self._output(model)

if __name__ == '__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240314'
    train = pd.read_csv(pwd+'/train.csv')
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = ['ccb1','ace1','beta1','diuret1','age_s1','dig1','waso','vaso1','aai','hctz1','chol','rcrdtime','mcs_s1','mi2slp02','stloutp','ntg1','timebedp','stonsetp','trig','timest2','twuweh02','avsao2nh','hremt2p','avdnop4','ahremop','slplatp','timest1p']
    import dict_list as dl
    columns = dl.general_health_10
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns=[])

    # 모델 설계
    num_features = X_train.shape[1]
    model_def = mymodel(num_features, 0.3)
    model_def.build('DNN')
    dnn = model_def.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model = KerasClassifier(dnn, epochs=1000, batch_size=128, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    
    # grid 파라미터 지정
    epochs = [100,500,1000,2000,3000,5000,10000]
    batch_size = [16,32,64,128,256,512]
    dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    param_grid = dict(epochs=epochs, batch_size=batch_size, dropout=dropout, model__init_mode=init_mode)


    # grid 학습
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


    # # 모델 학습
    # epoch = 1000
    # batch_size = 128
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)      # DNN patience=5, CNN patience=10

    # history = model_def.model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
    # history = model.model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_val, y_val))

    # # XGBoost 모델
    # dnn, xgboost = model._XGBoost()
    # y_pred = xgboost.predict(dnn(X_test))

    # from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    # # 모델 평가
    # accu = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    # conf_mat = confusion_matrix(y_test, y_pred)
    # print(f'Accuracy: {accu}\nF1 Score: {f1}')
    # print("Confusion Matrix:")
    # print(conf_mat)

    # # 모델 평가
    # acc = accuracy(history)
    # acc.run(X_test, y_test)

    # # print(model.model.summary())
    # # print(dnn.summary())