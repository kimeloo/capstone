import pandas as pd
from preprocess import run
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from keras.regularizers import l1

if __name__ == '__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240314'
    train = pd.read_csv(pwd+'/train.csv')
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    columns = ['ccb1','ace1','beta1','diuret1','age_s1','dig1','waso','vaso1','aai','hctz1']
    X_train, y_train, X_val, y_val, X_test, y_test = run(train, val, test, strategy, columns)

    # 모델 설계
    num_features = X_train.shape[1]
    model = Sequential([
        Dense(256, activation='relu', input_shape=(num_features,), kernel_regularizer=l1(0.0001)),   # DNN 개선
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # 모델 학습
    epoch = 1000
    batch_size = 128
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
    print(model.summary())
    
    # 모델 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test loss: {test_loss}\nTest accuracy: {test_acc}')

    # F1 score
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    # 정확도 그래프
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    # 오차 그래프
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # confusion matrix
    plt.subplot(1,3,3)
    conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    plt.imshow(conf_mat_norm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf_mat[i, j]), ha='center', va='center', fontsize=20, color='red')
    plt.show()