import tensorflow as tf
import keras

# mnist data 불러오기, 학습/테스트 데이터로 분리
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 학습 데이터에서 검증 데이터 분리 (6만개 중 1만개)
x_val = x_train[50000:]
x_train = x_train[:50000]
y_val = y_train[50000:]
y_train = y_train[:50000]

# 데이터 구조 변경 : 2차원 -> 3차원(색상 정보 추가)
import numpy as np
x_train = np.reshape(x_train, (50000, 28, 28, 1))
x_val = np.reshape(x_val, (10000, 28 ,28 ,1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

# 데이터 정규화 (0~255) -> (0~1)
x_train = x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# y 값을 one hot encoding으로 변경
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# CNN 텐서플로 구현
model = keras.Sequential([
    keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# 모델 평가
test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)
