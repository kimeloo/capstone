import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# CSV 파일 불러오기
pwd = '/Users/kimeloo/Documents/Coding/capstone/data/240201'
# train_file = 'diab_train_data_replaced.csv'
data_file = 'test_03_psMatching_htnderv_s1.csv'
data = pd.read_csv(f'{pwd}/{data_file}')


columns = ['oai0p', 'oai4p', 'oai4pa', 'cai0p', 'cai4p', 'cai4pa']
for i in ['0', '2', '3', '4', '5']:
    for j in ['p', 'pa', 'ps', 'pns']:
        columns.append(f'rdi{i}{j}')
    columns.append(f'rdirem{i}p')
    columns.append(f'rdinr{i}p')

# 변수 분리
X = data.drop(['htnderv_s1', 'PropensityScore'], axis=1).copy()
X = data[columns].copy()
y = data['htnderv_s1']

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# 데이터 정규화
scaler = StandardScaler()
X_train_fit = scaler.fit_transform(X_train)
X_test_fit = scaler.transform(X_test)

# 모델 설계
num_features = X_train_fit.shape[1]
dropout = 0.3
model = Sequential([
    Dense(256, activation='relu', input_shape=(num_features,)),
    Dropout(dropout),
    Dense(256, activation='relu'),
    Dropout(dropout),
    Dense(256, activation='relu'),
    Dropout(dropout),
    Dense(256, activation='relu'),
    Dropout(dropout),
    Dense(128, activation='relu'),
    Dropout(dropout),
    Dense(64, activation='relu'),
    Dropout(dropout),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 모델 학습
epoch = 550
# history = model.fit(X_train_fit, y_train, epochs=epoch, validation_split=0.2, callbacks=[early_stopping])
history = model.fit(X_train_fit, y_train, epochs=epoch, validation_split=0.2)

# 모델 평가
test_loss, test_acc = model.evaluate(X_test_fit, y_test)
print(f'Test loss: {test_loss}\nTest accuracy: {test_acc}')

# F1 score
y_pred = (model.predict(X_test_fit) > dropout).astype("int32")
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
        plt.text(j, i, str(conf_mat[i, j]), ha='center', va='center', color='red')
plt.show()