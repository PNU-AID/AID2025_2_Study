# Chapter 03-3

1. **다중 회귀**
    1. 2개 이상의 특성 사용 → 평면 학습 (**n차원**)
    2. 선형 회귀가 무조건 다중 회귀보다 성능이 낮은 것은 아님
2. **특성 공학**
    1. **기존의 특성을 사용**해 **새로운 특성**을 뽑아내는 작업
    2. ex) 농어 길이 * 농어 높이
3. **변환기**
    1. **특성을 만들거나 전처리**하기 위한 다양한 클래스
    2. fit(), transform() 제공
4. **규제**
    1. 머신러닝 모델이 훈련 세트를 **너무 과도하게 학습하지 못하도록** 막는 것
        
        ![image.png](image.png)
        
    2. **릿지(ridge)**
        1. 계수를 **제곱**한 값을 기준으로 규제 적용
    3. **라쏘(lasso)**
        1. 계수의 **절댓값**을 기준으로 규제 적용
        2. 계수가 **0이 될 확률** 높음 → 절댓값이기 때문 (릿지는 제곱이라 좀 더 완만 but 라쏘는 절댓값이라 뾰족한 그래프)
    4. 값 조정 : **alpha**
        1. alpha 값 **커짐** → 규제 커짐 → **과소적합** 유도
        2. alpha 값 **작아짐** → 규제 작아짐 → **과대적합** 유도
        3. **하이퍼파라미터** : **사람**이 알려줘야 하는 파라미터
        4. **훈련 세트, 테스트 세트 점수가 가장 가까운** 지점이 **최적의 alpha**
            
            ![이때의 최적 alpha : 10^-1 = 0.1](image%201.png)
            
            이때의 최적 alpha : 10^-1 = 0.1
            
5. 실습
    1. 1이 추가되는 이유 → 절편을 항상 값이 1인 특성과 곱해지는 계수로 보기 때문 but 사이킷런 선형 모델은 자동으로 절편 추가 → include_bias=False
    2. get_feature_names_out() : 특성이 각각 어떤 입력의 조합으로 만들어졌는지 알려줌
    3. 특성의 개수를 크게 늘리면 선형 모델 성능 향상 but 과대적합 → 테스트 세트 점수 매우 낮음
    4. 규제를 적용하기 위해서는 정규화 먼저 해야 함 → StandardScaler 사용
    - 코드
        
        ```python
        import pandas as pd
        perch_full = pd.read_csv('https://bit.ly/perch_csv_data')
        perch_full.head()
        ```
        
        ```python
        import numpy as np
        
        perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
               115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
               150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
               218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
               556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
               850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
               1000.0])
        ```
        
        ```python
        from sklearn.model_selection import train_test_split
        
        train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
        ```
        
        ```python
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(include_bias=False) # False 하면 1 사라짐
        poly.fit([[2, 3]])
        print(poly.transform([[2, 3]]))
        ```
        
        ```python
        poly = PolynomialFeatures(include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        train_poly.shape
        ```
        
        ```python
        poly.get_feature_names_out()
        ```
        
        ```python
        test_poly = poly.transform(test_input)
        ```
        
        ```python
        from sklearn.linear_model import LinearRegression
        
        lr = LinearRegression()
        lr.fit(train_poly, train_target)
        
        print(lr.score(train_poly, train_target))
        print(lr.score(test_poly, test_target))
        ```
        
        ```python
        poly = PolynomialFeatures(degree=5, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
        print(train_poly.shape)
        ```
        
        ```python
        lr.fit(train_poly, train_target)
        print(lr.score(train_poly, train_target))
        print(lr.score(test_poly, test_target)) # 음수 -> 과대적합
        ```
        
        ```python
        from sklearn.preprocessing import StandardScaler
        
        ss = StandardScaler()
        ss.fit(train_poly)
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)
        ```
        
        ```python
        from sklearn.linear_model import Ridge
        
        ridge = Ridge()
        ridge.fit(train_scaled, train_target)
        print(ridge.score(train_scaled, train_target))
        print(ridge.score(test_scaled, test_target))
        ```
        
        ```python
        import matplotlib.pyplot as plt
        
        train_score = []
        test_score = []
        ```
        
        ```python
        alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
        
        for alpha in alpha_list:
            ridge = Ridge(alpha=alpha)
            ridge.fit(train_scaled, train_target)
            train_score.append(ridge.score(train_scaled, train_target))
            test_score.append(ridge.score(test_scaled, test_target))
        ```
        
        ```python
        plt.plot(alpha_list, train_score)
        plt.plot(alpha_list, test_score)
        plt.xscale('log')
        plt.xlabel('alpha')
        plt.ylabel('R^2')
        plt.show()
        ```
        
        ```python
        ridge = Ridge(alpha=0.1)
        ridge.fit(train_scaled, train_target)
        print(ridge.score(train_scaled, train_target))
        print(ridge.score(test_scaled, test_target))
        ```
        
        ```python
        np.sum(ridge.coef_ == 0)
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
        ```
        
        ```python
        plt.plot(alpha_list, train_score)
        plt.plot(alpha_list, test_score)
        plt.xscale('log')
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
        np.sum(lasso.coef_ == 0) # 40개의 특성을 0으로 만듦
        ```
        
    
    [03_3.ipynb](03_3.ipynb)