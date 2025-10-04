# Chapter 3-1

- 지도학습 알고리즘
    1.  분류
        1. 샘플을 몇개의 클래스 중 하나로 분류하는 문제, 
            
            ex) 도미와 빙어의 길이 무게를 통해 새로운 값이 도미인지 빙어인지 분류하는 모델(chapter2)
            
        2. k-최근접 이웃 알고리즘에서는 KNeighborsClassifier를 사용
    2. 회귀
        1. 정해진 클래스 없이 임의의 수치를 출력
        2. 이상적인 값을 내는것이 아닌 연속적인 값을 구하는 문제에 적합
        3. 두 변수 사이의 상관관계를 분석하는 방법
    
- 1차원 배열을 2차원 배열로 변환
    1. chapter2에서는 column_stack을 이용하여 길이, 무게로 묶인 2차원배열을 만들었다.
    2. chapter3에서는 농어의 테스트 세트 길이와 무게를 기반으로 농어의 무게를 찾아내는 모델을 만들려고 하기 때문에 무게는 target으로 들어가게된다.
    3. input인 length는 1차원배열이기 때문에 numpy의 reshape기능을 사용하여 2차원배열로 변환
    
    ```python
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
    
    print(train_input.shape, test_input.shape)
    ```
    

- 결정계수(R² )
    1. KNeighborsRegressor()를 사용했을때 knr.score의 결과 (분류모델에서는 정확도라고 평가함)
        1. **$R^2=1-\frac{(타겟-예측)^2의\space합}{(타겟-평균)^2의\space합}$**
        <br>
        <img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/79527e87-2aa6-42e9-97cc-cbe7518951fa" />

        
    2. 과대적합: 훈련세트의 결정계수는 잘 나왔으나 테스트세트의 결정계수는 낮게 나온 경우
        1. 훈련세트에만 잘 맞는 모델이라 테스트세트 사용시 새로운 샘플에 대한 예측이 잘 동작하지 않는 경우를 말한다.
    3. 과소적합: 훈련세트의 결정계수가 테스트세트의 결정계수보다 작은경우
        1. 모델이 너무 단순하여 훈련세트에 적절히 훈련되지 않은 경우를 말한다.
        2. 해결방안: 모델을 더 복잡하게 만든다
            
            ex) k-최근접 이웃 회귀 알고리즘 사용시에 이웃의 개수를 줄인다
            
            이웃의 개수를 줄이면 국지적인 패턴에 민감해지고 이웃의 개수를 늘리면 데이터 전반에 있는 일반적인 패턴을 따를것이다.
            
    
- Chapter 3-1 python code
    
    ```python
    import numpy as np
    #길이 데이터를 통해 무게 예측하기
    #target -> weight
    perch_length = np.array(
        [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
         21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
         22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
         27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
         36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
         40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
         )
    perch_weight = np.array(
        [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
         110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
         130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
         197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
         514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
         820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
         1000.0, 1000.0]
         )
    import matplotlib.pyplot as plt
    plt.scatter(perch_length, perch_weight)
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    ```
    
    ```python
    from sklearn.model_selection import train_test_split
    
    train_input, test_input, train_target, test_target = train_test_split(
        perch_length, perch_weight, random_state=42)
        
    #1차원배열인 perch_length를 2차원 배열로 만들어주는 작업
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
    
    print(train_input.shape, test_input.shape)
    ```
    
    ```python
    from sklearn.neighbors import KNeighborsRegressor
    #회귀모델 사용하기
    knr = KNeighborsRegressor()
    #k-최근접 이웃 회귀모델 훈련
    knr.fit(train_input, train_target)
    
    #테스트세트의 결정계수
    print(knr.score(test_input, test_target))
    #결과값: 0.992809...
    ```
    
    ```python
    from sklearn.metrics import mean_absolute_error
    #타킷과 예측사이의 절댓값 오차를 평균하여 반환
    test_prediction = knr.predict(test_input)
    mae = mean_absolute_error(test_target, test_prediction)
    print(mae)
    
    #훈련세트의 결정계수
    print(knr.score(train_input, train_target))
    #결과값: 0.969882...
    #테스트세트의 결정계수가 훈련세트의 결정계수보다 큼 -> 과소적합
    ```
    
    ```python
    #과소적합을 해결하기 위해 이웃의 개수를 줄임
    knr.n_neighbors = 3
    knr.fit(train_input, train_target)
    print(knr.score(train_input, train_target))
    print(knr.score(test_input, test_target))
    #결과값-> 훈련세트: 0.980489... 테스트세트: 0.974645...
    ```
