# Chapter 1

- Chapter 1-3 python code
    
    ```python
    #도미 데이터
    bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
    bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
    
    import matplotlib.pyplot as plt
    
    plt.scatter(bream_length, bream_weight)
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    #빙어 데이
    smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
    smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
    
    plt.scatter(bream_length, bream_weight) #산점도
    plt.scatter(smelt_length, smelt_weight)
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    --------------------------------------------------------------------------------
    length = bream_length+smelt_length
    weight = bream_weight+smelt_weight
    
    fish_data = [[l, w] for l, w in zip(length, weight)]
    
    print(fish_data)
    
    fish_target = [1]*35 + [0]*14 #머신러닝을 위한 이진분류
    print(fish_target)
    --------------------------------------------------------------------------------
    from sklearn.neighbors import KNeighborsClassifier #k-최근접 이웃 알고리즘을 위한 클래스
    kn = KNeighborsClassifier()
    
    kn.fit(fish_data, fish_target) #주어진 데이터로 알고리즘 훈련
    kn.score(fish_data, fish_target) #정확도 확인
    --------------------------------------------------------------------------------
    #k-최근접 이웃 알고리즘
    plt.scatter(bream_length, bream_weight) 
    plt.scatter(smelt_length, smelt_weight)
    plt.scatter(30, 600, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    kn.predict([[30, 600]]) #값을 통해 어떤 array에 분류될지 예측
    
    print(kn._fit_X)
    print(kn._y)
    
    kn49 = KNeighborsClassifier(n_neighbors=49)
    kn49.fit(fish_data, fish_target)
    kn49.score(fish_data, fish_target)
    ```
    

## 머신러닝이란?

규칙을 일일이 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야

대표 라이브러리: 사이킷런([scikit-learn]
## 딥러닝이란?

인공신경망을 기반으로 한 머신러닝 분야

대표 라이브러리: 텐서플로(TensorFlow), 파이토치(PyTorch)

k-최근접 이웃알고리즘

주변의 가장 가까운 K개의 데이터를 보고 데이터가 속할 그룹을 판단하는 알고리즘이 K-NN 알고리즘이다.
