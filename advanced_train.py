from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Fashion MNIST 데이터 다운 및 시각화
# 데이터 다운
fashion_data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_data.load_data()

# 시각화
plt.imshow(train_images[0])
plt.title(class_names[train_labels[0]])
plt.show()

# 데이터 가공
train_images = train_images / 255.0
test_images = test_images / 255.0

# 인공신경망 구현 및 컴파일
# 모델 구현
# [28, 28] -> [28 * 28] -> [128] -> [10]

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))  # 2차원 -> 1차원으로 변경
model.add(keras.layers.Dense(128, activation='relu'))  # 128개 노드
model.add(keras.layers.Dense(10, activation='softmax'))  # result

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

# 모델 학습 및 테스트
epoch = 20
batch_size = 16

# 학습
model.fit(train_images, train_labels, epochs=epoch,
          batch_size=batch_size, verbose=2)

# 테스트(예측)
predict_labels = model.predict(test_images)

# 테스트 결과 시각화
# 랜덤으로 데이터 10개 선택
test_list = list(range(0, len(test_images)))
random_image = []
for i in range(10):
    append_image = test_list.pop(test_list.index(random.choice(test_list)))
    random_image.append(append_image)

# 데이터 시각화
num_rows = 2
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


for i in range(10):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(random_image[i], predict_labels, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(random_image[i], predict_labels, test_labels)

plt.show()
