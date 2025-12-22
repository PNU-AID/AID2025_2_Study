# 08-3 | 합성곱 신경망의 시각화

### 가중치 시각화

- 코드
    
    ```python
    # 8-2에서 만든 모델 불러오기
    import keras
    model = keras.models.load_model('best-cnn-model.keras')
    # 층 출력
    model.layers
    # 첫번째 합성곱 층의 가중치
    conv = model.layers[0]
    print(conv.weights[0].shape, conv.weights[1].shape)
    # 텐서플로의 다차원 배열->넘파이 배열로 변환
    conv_weights = conv.weights[0].numpy()
    print(conv_weights.mean(), conv_weights.std())
    # 히스토그램
    import matplotlib.pyplot as plt
    plt.hist(conv_weights.reshape(-1, 1)) # 히스토그램은 1차원 배열이여야함
    plt.xlabel('weight')
    plt.ylabel('count')
    plt.show()
    # 커널 출력
    fig, axs = plt.subplots(2, 16, figsize=(15,2))
    for i in range(2):
        for j in range(16):
            axs[i, j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
            axs[i, j].axis('off')
    plt.show()
    
    # 훈련하지않은 빈 합성곱 신경망
    no_training_model = keras.Sequential()
    no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                              padding='same', input_shape=(28,28,1)))
    # 가중치 저장
    no_training_conv = no_training_model.layers[0]
    print(no_training_conv.weights[0].shape)
    # 넘파이 배열로 변환
    no_training_weights = no_training_conv.weights[0].numpy()
    print(no_training_weights.mean(), no_training_weights.std())
    # 히스토그램
    plt.hist(no_training_weights.reshape(-1, 1))
    plt.xlabel('weight')
    plt.ylabel('count')
    plt.show()
    # 커널 출력
    fig, axs = plt.subplots(2, 16, figsize=(15,2))
    for i in range(2):
        for j in range(16):
            axs[i, j].imshow(no_training_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
            axs[i, j].axis('off')
    plt.show()
    ```
    

### 함수형 API

- 코드
    
    ```python
    # 함수형 API
    
    # 7장에서 만든 인공 신경망
    inputs = keras.Input(shape=(784,))
    dense1 = keras.layers.Dense(100, activation='relu')
    dense2 = keras.layers.Dense(10, activation='softmax')
    # 을 API로 구현
    hidden = dense1(inputs)
    outputs = dense2(hidden)
    func_model = keras.Model(inputs,outputs)
    print(model.inputs)
    # 입력과 Conv2D 출력을 연결
    # 다음 객체의 predict를 호출하면 첫번째 conv2D의 출력을 반환할 것임.
    conv_acti = keras.Model(model.inputs, model.layers[0].output)
    ```
    

![image.png](image.png)

Q. 특성 맵 시각화를 만드는데 함수형 API가 필요한 이유?

A.  중간층의 출력값을 직접 꺼내오기 위해서

특성 맵 시각화=중간층 출력 시각화

Sequential 모델에서는

**중간층의 출력만 따로 빼서 모델처럼 만들기가 불편하거나 거의 불가능**

반면 함수형 API는

레이어의 입력과 출력을 노드처럼 다루기 때문에

```
입력 → [특정 중간층] → 출력
```

이런 경로를 **새로운 모델로 만들 수 있음.**

### 특성 맵 시각화

- 코드
    
    ```python
    # 특성 맵 시각화
    
    # 첫번째 샘플 그리기
    (train_input, train_target), (test_input, test_target) = \
      keras.datasets.fashion_mnist.load_data()
    plt.imshow(train_input[0], cmap='gray_r')
    plt.show()
    # Conv2D층이 만드는 특성 맵 출력
    inputs = train_input[0:1].reshape(-1, 28, 28, 1)/255.0
    feature_maps = conv_acti.predict(inputs)
    print(feature_maps.shape)
    fig, axs = plt.subplots(4, 8, figsize=(15,8))
    for i in range(4):
        for j in range(8):
            axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
            axs[i, j].axis('off')
    plt.show()
    # 두번째 합성곱 층이 만든 특성 맵 출력
    conv2_acti = keras.Model(model.inputs, model.layers[2].output)
    feature_maps = conv2_acti.predict(train_input[0:1].reshape(-1, 28, 28, 1)/255.0)
    print(feature_maps.shape)
    fig, axs = plt.subplots(8, 8, figsize=(12,12))
    for i in range(8):
        for j in range(8):
            axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
            axs[i, j].axis('off')
    plt.show()
    ```
    

![첫번째 층의 특성 맵 시각화 32개](image%201.png)

첫번째 층의 특성 맵 시각화 32개

![ 두번째 층의 특성 맵 시각화 결과. 64개](image%202.png)

 두번째 층의 특성 맵 시각화 결과. 64개

합성곱 층을 많이 쌓을수록 직관적으로 이해하기 어려움

→ 합성곱 신경망의 앞부분 층은 이미지의 시각적인 정보를 감지하고

뒷부분 층은 시각적인 정보를 바탕으로 추상적인 정보를 학습한다고 볼 수 있음.