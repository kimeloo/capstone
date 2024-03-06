import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'test_01_preprocess.csv'
# filename = 'diab_train_data_replaced.csv'

# DataFrame 생성
X = pd.read_csv(f'{pwd}/{filename}').drop(columns='parrptdiab')
y = pd.read_csv(f'{pwd}/{filename}')['parrptdiab']

# X에서 원하는 컬럼 지정
## 원하는 전체
# column_list = 'rdi0p, rdi2p, rdi3p, rdi4p, rdi5p, rdi0pa, rdi2pa, rdi3pa, rdi4pa, rdi5pa, rdirem0p, rdirem2p, rdirem3p, rdirem4p, rdirem5p, rdinr0p, rdinr2p, rdinr3p, rdinr4p, rdinr5p, oai0p, oai4p, oardrbp, oardrop, oardnbp, oardnop, cai0p, cai4p, cardrbp, cardrop, cardnbp, cardnop, pctstapn, remepbp, remepop, remlaip, remlaiip, ahrembp, ahremop, ahnrembp, ahnremop, hremt1p, hremt2p, hremt34p, timebedp, slpprdp, slpeffp, slplatp'
## 원하는 전체 중 상위 일부
column_list = 'timebedp, remlaip, slpprdp, slpeffp, ahnremop, remlaiip, rdirem0p, rdinr0p, rdi0p, rdi0pa, pctstapn, ahremop, ahnrembp'

# column_list의 값을 ', '로 split
column_list = column_list.split(', ')

# column_list의 컬럼만 남기기
X = X[column_list]

# 데이터 정규화
scaler = StandardScaler()
X_fit = scaler.fit_transform(X)

####################################################################################################

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# DNN 모델 생성
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy :", accuracy)

# 예측
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy :", accuracy)
print("F1-Score :", f1)

# 분류 보고서 출력
print(classification_report(y_test, y_pred))

# confusion matrix 계산
cm = confusion_matrix(y_test, y_pred)

# confusion matrix 그래프 출력
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})  # annot_kws를 사용하여 텍스트 크기 조절
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('parrptdiab')

plt.show()
