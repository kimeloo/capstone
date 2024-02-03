import pandas as pd
# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data'
filename_x_train = '240122/preprocessed_data.csv'
filename_y_train = '240122/dataset_230122.csv'
filename_xy_test = '240201/shhs1-dataset-0.20.0.csv'
# SHHS1에는 pd(mhpark) 컬럼이 없음
# SHHS2에 prknsn2 컬럼이 존재
# 이하는 각 컬럼 매칭
# # cvtia -> stroke15
# # poremlat -> remlaip
# # potmrem -> timeremp
# # pqpenth -> emacls25, phacls25, ql204b, ql205b
# # poremli -> remlaip? MrOS에 없음
# # m1adepr -> ntca1 (유사)
# # poremlii -> remlaiip
# # fosofam1 -> hlthlm25, probsa25(반대방향 응답)
# # ——
# # popqueeg2
# # pqblegs
# # polsao2r
# # bpbpsys2

# # pd(mhpark) -> prknsn2
# # cvcer, poplmrem -> 없음


# DataFrame 생성
X_train = pd.read_csv(f'{pwd}/{filename_x_train}')
y_train = pd.read_csv(f'{pwd}/{filename_y_train}')['pd']
X_test = pd.read_csv(f'{pwd}/{filename_xy_test}')


# 아래는 SHHS 데이터셋에 맞게 수정한 것
X_train = X_train[['cvtia', 'poremlat', 'potmrem', 'pqpenth', 'poremli', 'm1adepr', 'poremlii', 'fosofam1', 'poqueeg2', 'pqbplegs', 'polsao2r', 'bpbpsys2']].copy()
X_test = X_test[['stroke15', 'remlaip', 'timeremp', 'phacls25', 'remlaip', 'ntca1', 'remlaiip', 'hlthlm25', 'remlaip', 'remlaip', 'remlaip', 'remlaip']].copy()

# X_train의 컬럼에 맞게 X_test의 컬럼을 수정
# X_test.columns = ['cvtia', 'poremlat', 'potmrem', 'pqpenth', 'poremli', 'm1adepr', 'poremlii', 'fosofam1']
X_test.columns = ['cvtia', 'poremlat', 'potmrem', 'pqpenth', 'poremli', 'm1adepr', 'poremlii', 'fosofam1', 'poqueeg2', 'pqbplegs', 'polsao2r', 'bpbpsys2']

# 데이터 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_fit = scaler.fit_transform(X_train)
X_test_fit = scaler.fit_transform(X_test)
X_train = X_train_fit
X_test = X_test_fit


# -------------------
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix


# 모델 설계
num_features = X_train.shape[1]
model = Sequential([
    Dense(256, activation='relu', input_shape=(num_features,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 모델 학습
epoch = 550
# history = model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=epoch, validation_split=0.2, callbacks=[early_stopping])
history = model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=epoch, validation_split=0.1)

# 만들어진 모델에 맞춰 X_test의 예측값을 구함
y_pred = (model.predict(np.expand_dims(X_test, axis=-1)) > 0.5).astype("int32")

# 예측 값의 분포 확인
print(pd.Series(y_pred.flatten()).value_counts())

# # 모델 평가
# test_loss, test_acc = model.evaluate(np.expand_dims(X_test, axis=-1), y_test)
# print(f'Test loss: {test_loss}\nTest accuracy: {test_acc}')

# # F1 score
# y_pred = (model.predict(np.expand_dims(X_test, axis=-1)) > 0.5).astype("int32")
# f1 = f1_score(y_test, y_pred)
# print(f'F1 Score: {f1}')

# # Confusion matrix
# conf_mat = confusion_matrix(y_test, y_pred)

# # 정확도, 오차, confusionmatrix 그리기
# import matplotlib.pyplot as plt
# plt.figure(figsize = (10,5))
# epoch_range = np.arange(1, epoch + 1)
# # 정확도 그래프
# plt.subplot(1, 3, 1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['Training', 'Validation'], loc='upper left')
# plt.title('Accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# # 오차 그래프
# plt.subplot(1, 3, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['Training', 'Validation'])
# plt.title('Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# # confusion matrix
# plt.subplot(1,3,3)
# conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
# plt.imshow(conf_mat_norm, cmap='Blues', interpolation='nearest')
# plt.colorbar()
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, str(conf_mat[i, j]), ha='center', va='center', color='red')
# plt.show()