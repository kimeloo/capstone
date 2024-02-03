import pandas as pd
# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data'
filename_x = 'preprocessed_data.csv'
filename_y = 'dataset_230122.csv'

# DataFrame 생성
X = pd.read_csv(f'{pwd}/{filename_x}')
y = pd.read_csv(f'{pwd}/{filename_y}')['pd']

# X = X[['mhpark', 'mhparkt', 'poremlat', 'cvcer', 'poremli', 'popcsa95', 'm1adepr', 'cvtia', 'pqpenth', 'fosogact', 'podsrem4', 'fosofam1', 'fosomot2', 'podsnr3', 'poremlii', 'pobpmmin', 'potmremp', 'poqueeg2', 'potmrem', 'fosoact2']].copy()
X = X[['mhpark', 'mhparkt', 'poremlat', 'cvcer', 'poremli', 'popcsa95', 'm1adepr', 'cvtia', 'pqpenth', 'fosogact']].copy()
# X = X[['mhpark', 'mhparkt']].copy()

# 데이터 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_fit = scaler.fit_transform(X)

# 훈련, 테스트 세트 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_fit, y, test_size=0.2, random_state=100)


# -------------------
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix


# 모델 설계
num_features = X_train.shape[1]
model = Sequential([
    Conv1D(256, kernel_size=2, activation='relu', input_shape=(num_features, 1)),
    Conv1D(128, kernel_size=2, activation='relu'),
    Conv1D(128, kernel_size=3, activation='relu'),
    Conv1D(128, kernel_size=3, activation='relu'),
    Conv1D(128, kernel_size=4, strides=2, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 모델 학습
epoch = 550
history = model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=epoch, validation_split=0.2)

# 모델 평가
test_loss, test_acc = model.evaluate(np.expand_dims(X_test, axis=-1), y_test)
print(f'Test loss: {test_loss}\nTest accuracy: {test_acc}')

# F1 score
y_pred = (model.predict(np.expand_dims(X_test, axis=-1)) > 0.5).astype("int32")
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1}')

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# 정확도, 오차, confusionmatrix 그리기
import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
epoch_range = np.arange(1, epoch + 1)
# 정확도 그래프
plt.subplot(1, 3, 1)
plt.plot(epoch_range, history.history['accuracy'])
plt.plot(epoch_range, history.history['val_accuracy'])
plt.legend(['Training', 'Validation'], loc='upper left')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
# 오차 그래프
plt.subplot(1, 3, 2)
plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
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