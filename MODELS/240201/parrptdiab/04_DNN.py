import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

# CSV 파일 불러오기
pwd = '/Users/kimeloo/Documents/Coding/capstone/data/240201'
train_file = 'diab_train_data_replaced.csv'
test_file = 'shhs1_replaced.csv'
train = pd.read_csv(f'{pwd}/{train_file}')
test = pd.read_csv(f'{pwd}/{train_file}')

# 아웃라이어 제거 전 컬럼
columns = ['diasbp', 'trig', 'savbnbh', 'davbnbh', 'oximet51', 'ai_rem', 'pcs_s1', 'davbnoh', 'savbnoh', 'avgsat']
# 아웃라이어 제거 후 컬럼
columns = ['psg_month', 'oardrop2', 'oardrbp2', 'oarbp2', 'cardnop2', 'canop2', 'cardnbp2', 'canbp2', 'cardrop2', 'carop2']

# 변수 분리
X_train = train[columns].copy()
y_train = train['parrptdiab']
X_test = test[columns].copy()
y_test = test['parrptdiab']

# 데이터 정규화
scaler = StandardScaler()
X_train_fit = scaler.fit_transform(X_train)
X_test_fit = scaler.transform(X_test)

# 모델 설계
num_features = X_train_fit.shape[1]
model = Sequential([
    Dense(256, activation='relu', input_shape=(num_features,)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

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
y_pred = (model.predict(X_test_fit) > 0.5).astype("int32")
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