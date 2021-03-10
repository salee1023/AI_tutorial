## 인공지능 맛보기 

> linear regression & data 전처리 & 이미지 학습을 실습합니다.

<br/>

### 01. 단순 선형 회귀 모델 구현

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.linear_model import LinearModel


# 데이터 불러오기
train_data = np.load(".\\datasets\\linear_train.npy")
test_x = np.load(".\\datasets\\linear_test_x.npy")


# tf 형식에 맞게 변환
x_data = np.expand_dims(train_data[:,0], axis=1)
y_data = train_data[:,1]


# 모델 생성
model = LinearModel(num_units=1)


# 최적화 함수, 손실함수와 모델 바인딩
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
			  loss=tf.keras.losses.MSE,
			  metrics=[tf.keras.metrics.MeanSquaredError()])


# 모델 학습
model.fit(x=x_data, 
		  y=y_data, 
		  epochs=10, 
		  batch_size=32)


# 모델 테스트
prediction = model.predict(x=test_x,
    					   batch_size=None)


# 결과 시각화
plt.scatter(x_data,y_data,s=5,label="train data")
plt.scatter(test_x,prediction,s=5,label="prediction data")
plt.legend()
plt.show()


# 모델 정리
model.summary()
```

<br/>

### 02. 데이터 전처리 및 시각화 

##### 02-1.  config.py  

자주 변경되는 세팅값을 한 곳에서 관리하고 가져올 수 있도록 `config.py` 파일을 구현합니다.

![image-20210304173229967](README.assets/image-20210304173229967.png)

👉 현재는 경로만 설정해주었지만, 알고리즘 구현에 필요한 다른 변수 (epoch, batch size 등) 을 지정할 때도 유용하게 사용됨을 알게되었습니다. 

👉 사용법 : `config.py` 에서 `config`  클래스를 import 합니다.  `args = config.parser.parse_args()` 로 인자들을 파싱하여 `args` 에 저장합니다. 이후, `args.인자이름` 으로 사용하고싶은 인자값을 받을 수 있습니다.

<br/>

##### 02-2. train.py & preprocess.py

이미지 캡셔닝용 데이터를 전처리하는 `train.py` 와 `preprocess.py` 를 구현합니다. 

- **train.py**

![image-20210304174625419](README.assets/image-20210304174625419.png)

👉 저장된 데이터셋을 불러올 때는 따로 입력값을 받아 해당 데이터셋을 불러올 수 있도록 했습니다.

- **preprocess.py**

![image-20210304175257843](README.assets/image-20210304175257843.png)

👉 훈련용은 80%, 테스트용은 20%의 비율로 지정했습니다. 

👉 이미지 캡셔닝 데이터가 비슷한 데이터끼리 모여있었기 때문에 `랜덤`으로 데이터를 추출해서 데이터셋을 분리했더라면 더 객관적인 데이터셋이 되었을 것입니다.  

![image-20210304175340533](README.assets/image-20210304175340533.png)

<br/>

##### 02-3. visualization

데이터를 시각화합니다. `train.py` 를 실행하면 아래와 같이 랜덤의 이미지와 캡션이 잘 나오는 것을 확인할 수 있습니다. 

![image-20210304175827466](README.assets/image-20210304175827466.png)

<br/>

### 02. 인공신경망으로 옷 구별하기

> Fashion MNIST 데이터를 학습시키는 인공신경망을 구현합니다.

```python
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

# 분류할 객체들의 이름을 지정한다
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Fashion MNIST 데이터 다운
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

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 모델 학습 및 테스트
epoch = 20
batch_size = 16

# 학습
model.fit(train_images, train_labels, epochs=epoch,
          batch_size=batch_size, verbose=2)

# 테스트(예측)
predict_labels = model.predict(test_images)

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
```













