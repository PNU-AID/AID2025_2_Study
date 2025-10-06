# Chapter 03-1

1. **k-최근접 이웃 회귀 (k-nn)**
    1. **회귀** : 지도 학습 중 하나 → 임의의 어떤 **숫자를 예측** (↔ 분류)
    2. 예측하려는 샘플에 가장 **가까운 샘플 k개** 선택 → 이웃한 샘플한 타깃값의 평균 구함
2. **결정계수(R ** 2)** 
    1. 계산식 : **R ** 2 = 1 - (타깃 - 예측) ** 2의 합 / (타깃 - 평균) ** 2의 합**
    2. 예측이 **타깃에 가까워질수록 1**에 가까운 값이 됨
3. **과대적합** : 훈련 세트 >>> 테스트 세트
4. **과소적합** : 훈련 세트 <<< 테스트 세트 or 훈련 세트, 테스트 세트 모두 값 낮음
5. 실습
    1. 사이킷런에 사용할 훈련 세트 → 2차원 배열이어야 함 → 특성이 1개라면 수동으로 2차원 배열을 만들어야 함 (reshape 이용)
    2. 처음 과소적합 해결을 위해 k 값을 줄임 → k 값이 적을수록 더 가까이 있는 값의 평균이 나오기 때문에 훈련 세트의 값이 정확해짐
    - 코드
        
        ```python
        import numpy as np
        
        perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
               21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
               23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
               27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
               39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
               44.0])
        perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
               115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
               150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
               218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
               556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
               850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
               1000.0])
        ```
        
        ```python
        import matplotlib.pyplot as plt
        
        plt.scatter(perch_length, perch_weight)
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
        ```
        
        ```python
        from sklearn.model_selection import train_test_split
        
        train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42) # 훈련 세트, 테스트 세트 분리
        ```
        
        ```python
        test_array = np.array([1,2,3,4])
        print(test_array.shape)
        ```
        
        ```python
        test_array = test_array.reshape(2,2)
        print(test_array.shape) # 훈련하기 위해 2, 2로 reshape
        ```
        
        ```python
        test_array
        ```
        
        ```python
        train_input = train_input.reshape(-1, 1) # -1: 나머지 배열 크기에 맞춰라
        test_input = test_input.reshape(-1, 1)
        train_input.shape, test_input.shape
        ```
        
        ```python
        from sklearn.neighbors import KNeighborsRegressor
        
        knr = KNeighborsRegressor()
        
        knr.fit(train_input, train_target)
        knr.score(test_input, test_target) # 결정계수(R**2)
        ```
        
        ```python
        from sklearn.metrics import mean_absolute_error
        
        test_prediction = knr.predict(test_input)
        
        mae = mean_absolute_error(test_target, test_prediction)
        mae # 19g 정도 타깃값과 다름
        ```
        
        ```python
        knr.score(train_input, train_target) # 과소적합
        ```
        
        ```python
        knr.n_neighbors = 3 # 과소적합 해결 위함
        
        knr.fit(train_input, train_target)
        knr.score(train_input, train_target)
        ```
        
        ```python
        knr.score(test_input, test_target)
        ```
