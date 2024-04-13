from numpy import shape
import pandas as pd
from preprocess import run
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.optimizers.legacy import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.regularizers import l1

from keras.models import Model
import xgboost as xgb

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

def CNN_create(optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', input_shape=(X_shape, 1)))
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(epochs=epochs, optimizer=optimizer, loss=loss, metrics=metrics, batch_size=batch_size)
    return model

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
    X_shape = X_train.shape[1]

    # 모델 생성
    keras_model = KerasClassifier(model=CNN_create, epochs=100, batch_size=128, loss='binary_crossentropy', metrics=['accuracy'])

    epochs = [100, 200, 300, 400, 500, 1000, 2000, 3000, 5000, 10000]
    batch_size = [64, 128, 256, 512]
    loss = ['binary_crossentropy', 'categorical_crossentropy']
    metrics = ['accuracy']

    # grid 파라미터 지정
    param_grid = {
        'epochs': epochs,
        'batch_size': batch_size,
        'optimizer': [Adam(learning_rate=0.0001), RMSprop(learning_rate=0.0001)],
        'loss': loss,
        'metrics': metrics
    }

    # grid search
    grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))

    # 결과 출력
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} ({stdev}) with: {param}")
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")