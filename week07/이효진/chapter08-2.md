# 08-2 | 합성곱 신경망을 사용한 이미지 분류

패션 MNIST 데이터 불러오기

- 코드
    
    ```python
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    (train_input, train_target), (test_input, test_target) = \
        keras.datasets.fashion_mnist.load_data()
    # 일렬로 펼치지 않고 깊이 차원이 추가됨.
    train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
    train_scaled, val_scaled, train_target, val_target = train_test_split(
        train_scaled, train_target, test_size=0.2, random_state=42)
    ```
    

합성곱 신경망 만들기

- 코드
    
    ```python
    # 합성곱 신경망 만들기
    
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28,28,1)))
    # 첫번째 합성곱 층 Conv2D 추가
    # 32개의 필터, 커널크기 (3,3), 렐루 활성화 함수, 세임 패딩 사
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                  padding='same', input_shape=(28,28,1)))
    # 풀링 층 추가. 크기 (2,2)
    model.add(keras.layers.MaxPooling2D(2)) # 특성 맵 크기 (14,14,32)
    # 두번째 합성곱 층 & 풀링 층 추가
    # 64개의 필터
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu',
                                  padding='same'))
    model.add(keras.layers.MaxPooling2D(2)) # 특성 맵 크기 (7,7,64)
    # 밀집층 확률 계산을 위해 일렬로 펼침
    model.add(keras.layers.Flatten())
    # 바로 출력층으로 전달하지 않고 중간에 은닉층 추가
    model.add(keras.layers.Dense(100, activation='relu'))
    # 드롭아웃으로 은닉층의 과대적합 막음
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()
    # 층의 구성을 그림으로 표현해 주는 함수
    keras.utils.plot_model(model)
    keras.utils.plot_model(model, show_shapes=True)
    ```
    

모델 컴파일과 훈련

- 코드
    
    ```python
    # 모델 컴파일과 훈련 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',
                                                    save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                      restore_best_weights=True)
    history = model.fit(train_scaled, train_target, epochs=20,
                        validation_data=(val_scaled, val_target),
                        callbacks=[checkpoint_cb, early_stopping_cb])
    # 손실 그래프
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.show()
    # 검증 세트 성능 평가
    model.evaluate(val_scaled, val_target)
    ```
    

예측해보기

- 코드
    
    ```python
    # 훈련된 모델로 예측 해보기
    
    # 첫번째 샘플 이미지 확인
    plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
    plt.show()
    # 예측
    preds = model.predict(val_scaled[0:1])
    print(preds)
    # 막대 그래프로 표현
    plt.bar(range(1, 11), preds[0])
    plt.xlabel('class')
    plt.ylabel('prob.')
    plt.show()
    # 예측 결과 출력
    classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
               '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']
    import numpy as np
    print(classes[np.argmax(preds)])
    # 일반화 성능(테스트 세트에 대한 성능 측정)
    test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
    	model.evaluate(test_scaled, test_target)
    ```