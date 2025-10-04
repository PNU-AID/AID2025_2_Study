# Chapter 3-2

- k-최근접 이웃 회귀알고리즘의 한계
    1. 훈련세트의 범위에서 벗어난 샘플을 예측하고싶을때 값이 얼마나 커지든 같은 예측값을 보여준다
    2. k-최근접 이웃 회귀 알고리즘은 샘플의 이웃의 평균을 구하여 사용하기 때문에 아래 코드와 같이 길이가 50이 되든 100이 되든 1000이 되든 훈련세트의 범위에서 벗어났기 때문에 같은 이웃을 가지고 있으므로 같은 평균이 나온다.
    - 50, 100 산점도 비교 이미지


        <img width="580" height="432" alt="image" src="https://github.com/user-attachments/assets/93912747-a09f-409a-af5a-717f5a35d56c" />
        <img width="580" height="432" alt="image 1" src="https://github.com/user-attachments/assets/cd6cd198-54aa-41ff-9980-668d6f35d548" />


        

- 선형회귀 알고리즘
    1. 종속 변수 y와 한 개 이상의 독립 변수 (또는 설명 변수) X와의 선형 상관 관계를 모델링하는 회귀분석 기법
    2. 최소제곱법을 사용하여 주어진 샘플들에 가장 근사하는 직선을 나타내어 사용한다. 
        
        $MSE(w,b)=i=1∑n​(yi​−(wxi​+b))2$
        
    3. 해당 직선에 대한 가중치(기울기) 와 절편(편향)을 아래 코드로 구할 수 있다.
        
        ```python
        print(lr.coef_, lr.intercept_)
        ```
        
        1. 그러나 실습코드에서와 같이 절편이 음수로 나오면 안되지만 음수로 나오는 경우가 있다
        2. 위 값을 머신러닝 알고리즘이 찾은값을 모델 파라미터라고 한다. 최적의 모델파라미터를 찾는것을 모델 기반 학습이라고 하며, 훈련세트를 저장하는것이 훈련의 전부인것은 사례 기반 학습이라고 한다.
        3. 이는 최소제곱법을 이용하여 데이터상 가장 근사하게 만든 직선이기 때문에 데이터의 특성을 따로 고려하지 않아 생기는 문제이다.
        4. 위 문제를 해결하기 위해 다항회귀를 사용해보기로 하자.
            <img width="580" height="432" alt="image 2" src="https://github.com/user-attachments/assets/ab80671d-024f-44da-bd0c-3a067af2de9f" />

            

- 다항회귀
    1. 실습코드에 나오는 산점도를 보면 산점도의 형태가 직선이 아닌 곡선의 형태를 띄고있는것을 볼 수 있다.
    2. 이를 위해 1차식인 선형회귀 대신, 다항식을 사용하는 회귀인 다항회귀를 사용하면 더 정확한 모델을 만들 수 있다.
    3. 2차식을 위해 샘플들에 제곱인 column을 추가해준다.
        
        ```python
        train_poly = np.column_stack((train_input ** 2, train_input))
        test_poly = np.column_stack((test_input ** 2, test_input))
        ```
        
    4. 다항식을 이용하여 학습시킬때는 train_input이 아닌 다항을 포함하고있는 샘플인 train_poly를 사용하여 fit함수로 학습시킨다.
    5. 그러나 실습에서의 다항회귀를 사용하면 음수인 절편에 대한 문제는 사라져 훈련세트와 테스트세트의 결정계수는 높아졌으나 과소적합의 문제를 해결하지 못하였다.
    
    ```python
    print(lr.score(train_poly, train_target)) #결과값: 0.9706...
    print(lr.score(test_poly, test_target)) #결과값: 0.9775...
    #과소적합!
    ```
    
    <img width="580" height="432" alt="image 3" src="https://github.com/user-attachments/assets/305577e1-063a-4eb5-855d-eb5851e1b45e" />


- code
    
    ```python
    import numpy as np
    #데이터 준비
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
    ```
    
    ```python
    from sklearn.model_selection import train_test_split
    #훈련세트, 테스트세트 데이터 분리
    train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
    ```
    
    ```python
    #k-최근접 이웃 회귀 알고리즘 사용한 경우
    from sklearn.neighbors import KNeighborsRegressor
    knr = KNeighborsRegressor(n_neighbors=3)
    knr.fit(train_input, train_target)
    print(knr.predict([[50]])) #50의 길이를 가진 농어의 무게를 예측해보자
    #결과값: 1033 -> 실제로 잰 무게인 1.5키로와는 차이가 많이 남
    ```
    
    ```python
    import matplotlib.pyplot as plt
    
    distances, indexes = knr.kneighbors([[50]])
    
    plt.scatter(train_input, train_target)
    plt.scatter(train_input[indexes], train_target[indexes], marker='D')
    plt.scatter(50, 1033, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    print(np.mean(train_target[indexes]))
    #결과값: 1033 -> 이웃의 평균이 k-최근접 이웃 회구 알고리즘의 예측값으로 쓰인
    ```
    
    ```python
    #길이가 100일때의 산점도 및 예측
    import matplotlib.pyplot as plt
    distances, indexes = knr.kneighbors([[100]])
    
    plt.scatter(train_input, train_target)
    plt.scatter(train_input[indexes], train_target[indexes], marker='D')
    plt.scatter(100, 1033, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    print(knr.predict([[100]])) #길이가 100이여도 같은 값인 1033이 출력
    #훈련세트의 범위를 벗어난 샘플을 예측할때는 같은 무조건 같은 이웃을 가짐
    #이로인해 범위가 벗어났을때는 어떤 값을 가져도 같은 예측값이 나옴
    ```
    
    ```python
    #선형회귀 알고리즘
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    lr.fit(train_input, train_target)
    print(lr.predict([[50]])) #결과값: 1241.83...
    
    #가중치(기울기), 편향(절
    print(lr.coef_, lr.intercept_) #절편이 음수가 나옴
    ```
    
    ```python
    plt.scatter(train_input, train_target)
    
    plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
    
    plt.scatter(50, 1241.8, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    ```
    
    ```python
    print(lr.score(train_input, train_target)) #결과값: 0.93...
    print(lr.score(test_input, test_target)) #결과값: 0.82...
    ```
    
    ```python
    #다항회귀
    #2차방정식으로 나타내주기 위해 제곱을 column에 추가
    train_poly = np.column_stack((train_input ** 2, train_input))
    test_poly = np.column_stack((test_input ** 2, test_input))
    
    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    
    print(lr.predict([[50**2, 50]])) #결과값: 1573.98...
    print(lr.coef_, lr.intercept_)
    ```
    
    ```python
    # 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만든다.
    point = np.arange(15, 50)
    
    plt.scatter(train_input, train_target)
    
    plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
    
    plt.scatter([50], [1574], marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    print(lr.score(train_poly, train_target)) #결과값: 0.9706...
    print(lr.score(test_poly, test_target)) #결과값: 0.9775...
    ```
