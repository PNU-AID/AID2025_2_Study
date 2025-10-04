# Chapter 3-3

## 다중회귀

- 여러개의 특성을 사용한 선형회귀 모델
    1. chapter2에서처럼 1개의 특성을 사용하여 선형회귀 모델을 훈련시키면 모델의 그래프가 직선으로 나타남.
    2. 특성 2개인 선형회귀는 평면을 학습 → 특성 개수만큼의 차원을 학습하게 된다.
    3. 여러 개의 특성을 저장할 때 pandas의 데이터프레임에 저장하게 되면 간단하다.
        1. csv파일을 받아 pandas의 데이터 프레임으로 변환 후 이를 바로 입력 데이터로 사용 가능하다.
    

## 특성공학

- 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업
    1. ex) 3개의 특성이 주어졌을 때 각 특성을 서로 곱하는 형식으로 새로운 특성을 만들어낼 수 있다.

- 특성을 새로 만들거나 전처리 하기 위한 클래스를 변환기(transformer)라고 한다.
    1. ex) PolynomialFeatures
        1. fit()을 통해 새롭게 만들 특성 조합을 찾고 transform()을 통해 실제로 데이터를 변환한다.
        2. 변환기는 입력 데이터를 변환할 때 타깃데이터를 필요로 하지 않기 때문에 fit()에 입력 데이터만 전달한다.
        3. ex) 2개의 2, 3이라는 샘플이 있을때 샘플값, 샘플의 제곱값, 샘플끼리의 곱으로 특성개수를 늘린다.
            1. include_bias = False를 통해 무시되는 추가된 절편항을 추가 하지 않도록 하여 혼돈을 피한다.
            2. 특성을 더 추가하기 위해서는 매개변수 degree를 통해 필요한 고차항의 최대차수를 지정
            
            ```python
            poly = PolynomialFeatures(degree=5, include_bias=False)
            ```
            

- 특성의 개수가 너무 많아지게 된다면 완벽에 가까운 train set의 score를 얻을 수 있지만 test set의 score는 이상한 값이 나올 수 있다
    1. 이는 train set에 과대적합된 모델이므로 이를 조정하기 위해 규제를 사용한다.

## 규제

- 머신러닝 모델이 훈련세트에 과대적합되지 않도록 훼방하는 일
- 선형회귀의 경우에는 특성에 곱해지는 계수(기울기)의 크기를 작게 만든다. → 릿지, 라쏘 회귀
- 규제를 적용하기 위해서는 특성의 스케일을 통일시키는 정규화 작업을 해야한다.
    1. 사이킷런에서 제공하는 StandardScaler 클래스를 사용하면 간편하게 정규화가 가능하다.
- 매개변수 alpha를 이용하여 규제의 강도를 조절 할 수 있다.
    1. alpha의 값이 커지면 규제강도가 세지므로 계수의 값이 줄어들어 과소적합되도록 유도한다
    →과대적합인 모델에 적용하면 좋음
    2. alpha의 값이 작아지면 규제강도가 약해지므로 계수의 값이 커져 과대적합되도록 유도한다
    →과소적합인 모델에 적용하면 좋음

- 릿지 회귀
    
    ```python
    from sklearn.linear_model import Ridge
    
    ridge = Ridge()
    ridge.fit(train_scaled, train_target)
    ```
    
    1. 릿지를 통한 결과값을 혹인하면 train set의 score는 조금 낮아지지만 test set의 score가 정상 범주에 들어오게 된다.
    
- 라쏘 회귀
    
    ```python
    from sklearn.linear_model import Lasso
    
    lasso = Lasso()
    lasso.fit(train_scaled, train_target)
    ```
    
    1. 릿지 회귀와 마찬가지로 규제를 하지만 라쏘 회귀는 계수의 크기를 아예 0으로 만들 수도 있다.

- alpha 리스트를 만들어 변하는 alpha값마다의 score를 확인하여 최적의 alpha값을 찾아낼 수 있다.
- Chapter 3-3 python code
    
    ```python
    #pandas를 사용하여 데이터 읽어오기
    import pandas as pd
    
    df = pd.read_csv('https://bit.ly/perch_csv_data')
    perch_full = df.to_numpy()
    print(perch_full)
    ```
    
    ```python
    #데이터 받아서 훈련세트, 타킷, 테스트세트, 타킷 분리하기
    import numpy as np
    
    perch_weight = np.array(
        [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
         110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
         130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
         197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
         514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
         820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
         1000.0, 1000.0]
         )
    
    from sklearn.model_selection import train_test_split
    
    train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
    ```
    
    ```python
    #변환기의 사용예제 ->  특성을 추가해준다
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures()
    poly.fit([[2, 3]])
    print(poly.transform([[2, 3]]))
    
    poly = PolynomialFeatures(include_bias=False) #원래 절편항을 무시하지만 이해하기 편하도록
    poly.fit([[2, 3]])
    print(poly.transform([[2, 3]]))
    ```
    
    ```python
    #실제 데이터에 적용하여 특성 늘려주기
    poly = PolynomialFeatures(include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    print(train_poly.shape) #결과값:(42, 9)
    
    poly.get_feature_names_out() #어떤 특성이 어떤 입력의 조합으로 만들어졌는지 보여줌
    ```
    
    ```python
    test_poly = poly.transform(test_input)
    
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    
    print(lr.score(train_poly, train_target)) #결과값: 0.990318...
    print(lr.score(test_poly, test_target)) #결과값: 0.971455...
    #과소적합 문제 해결!
    ```
    
    ```python
    #최고차항을 지정하여 특성의 개수 더 늘려주기
    poly = PolynomialFeatures(degree=5, include_bias=False) 
    
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    test_poly = poly.transform(test_input)
    
    print(train_poly.shape)
    
    lr.fit(train_poly, train_target)
    print(lr.score(train_poly, train_target)) #결과값: 0.999999...
    print(lr.score(test_poly, test_target)) #결과값: -144.405792...
    #train set보다 많은 특성이 만들어져 과대적합이 됨
    ```
    
    ```python
    #간편하게 정규화해주기
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(train_poly)
    train_scaled = ss.transform(train_poly)
    test_scaled = ss.transform(test_poly)
    ```
    
    ```python
    from sklearn.linear_model import Ridge #릿지 회귀를 통한 규제
    
    ridge = Ridge()
    ridge.fit(train_scaled, train_target)
    print(ridge.score(train_scaled, train_target)) #결과값: 0.989610...
    print(ridge.score(test_scaled, test_target)) #결과값: 0.979069...
    #과대적합 문제 해결 but test set의 점수가 증가하지 않음 -> 최적의 alpha값을 찾아 적용
    ```
    
    ```python
    #가장 적합한 alpha값 찾기
    import matplotlib.pyplot as plt
    
    train_score = []
    test_score = []
    
    alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
    for alpha in alpha_list:
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_scaled, train_target)
        train_score.append(ridge.score(train_scaled, train_target))
        test_score.append(ridge.score(test_scaled, test_target))
        
    plt.plot(np.log10(alpha_list), train_score)
    plt.plot(np.log10(alpha_list), test_score)
    plt.xlabel('alpha')
    plt.ylabel('R^2')
    plt.show()
    ```
    
    ```python
    ridge = Ridge(alpha=0.1) #가장 적합한 alpha값:0.1
    ridge.fit(train_scaled, train_target)
    
    print(ridge.score(train_scaled, train_target)) #결과값: 0.989789...
    print(ridge.score(test_scaled, test_target)) #결과값: 0.980059...
    ```
    
    ```python
    from sklearn.linear_model import Lasso
    
    lasso = Lasso()
    lasso.fit(train_scaled, train_target)
    print(lasso.score(train_scaled, train_target))
    print(lasso.score(test_scaled, test_target))
    ```
    
    ```python
    train_score = []
    test_score = []
    
    alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
    for alpha in alpha_list:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(train_scaled, train_target)
        train_score.append(lasso.score(train_scaled, train_target))
        test_score.append(lasso.score(test_scaled, test_target))
        
    plt.plot(np.log10(alpha_list), train_score)
    plt.plot(np.log10(alpha_list), test_score)
    plt.xlabel('alpha')
    plt.ylabel('R^2')
    plt.show()
    ```
    
    ```python
    lasso = Lasso(alpha=10)
    lasso.fit(train_scaled, train_target)
    
    print(lasso.score(train_scaled, train_target))
    print(lasso.score(test_scaled, test_target))
    ```
    
    ```python
    print(np.sum(lasso.coef_ == 0)) #계수가 0인 특성의 개수
    ```
