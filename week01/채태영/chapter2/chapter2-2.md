# Chapter2-2

- Chapter 2-2 python code
    
    ```python
    fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                    31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                    35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                    10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
    fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                    500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                    700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                    7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
    
    import numpy as np
    np.column_stack(([1, 2, 3], [4, 5, 6]))
    #for문과 zip을 사용하여 행별로 원소를 꺼내는 작업의 간편화 -> column_stack
    fish_data = np.column_stack((fish_length, fish_weight))
    print(fish_data[:5])
    print(np.ones(5))
    
    fish_target = np.concatenate((np.ones(35), np.zeros(14)))
    print(fish_target)
    --------------------------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    #2-1에서처럼 random모듈을 사용하는 대신 아래 코드와 같이 한번에 훈련세트와 테스트세트를 나눌 수 있다.
    train_input, test_input, train_target, test_target = train_test_split(
        fish_data, fish_target, random_state=42)
    print(train_input.shape, test_input.shape)
    print(train_target.shape, test_target.shape)
    print(test_target)
    
    #stratify를 사용해 target에 있는 비율대로 샘플링을 할 수 있다.
    train_input, test_input, train_target, test_target = train_test_split(
        fish_data, fish_target, stratify = fish_target, random_state=42)
    print(test_target)
    -------------------------------------------------------------------------------------
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    kn.fit(train_input, train_target)
    kn.score(test_input, test_target)
    print(kn.predict([[25, 150]]))
    
    import matplotlib.pyplot as plt
    plt.scatter(train_input[:,0], train_input[:,1])
    plt.scatter(25, 150, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    -------------------------------------------------------------------------------------
    #데이터 전처리가 되지 않았을때의 예시코드
    distances, indexes = kn.kneighbors([[25, 150]])
    
    plt.scatter(train_input[:,0], train_input[:,1])
    plt.scatter(25, 150, marker='^')
    plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    print(train_input[indexes])
    print(train_target[indexes])
    
    plt.scatter(train_input[:,0], train_input[:,1])
    plt.scatter(25, 150, marker='^')
    plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
    plt.xlim((0, 1000))
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    -------------------------------------------------------------------------------------
    #표준편차를 기준으로 데이터전처리를 했을때의 예시코드
    mean = np.mean(train_input, axis = 0)
    std = np.std(train_input, axis = 0)
    
    train_scaled = (train_input - mean) / std
    
    plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
    plt.scatter(25, 150, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show
    
    new = ([25, 150] - mean) / std
    
    plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
    plt.scatter(new[0], new[1], marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show
    
    kn.fit(train_scaled, train_target)
    test_scaled = (test_input - mean) / std
    kn.score(test_scaled, test_target)
    
    print(kn.predict([new]))
    
    distances, indexes = kn.kneighbors([new])
    plt.scatter(train_scaled[:,0], train_scaled[:,1])
    plt.scatter(new[0], new[1], marker='^')
    plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    ```
    
- 스케일
    1. 파이썬코드의 데이터와 같이 주어진 데이터가 길이와 무게로 서로의 단위도 다르며 두 특성의 범위도 다르다. 이를 두 특성의 스케일(scale)이 다르다고 한다.
    2. 데이터 전처리과정에서 이러한 스케일이 다른 데이터의 일정한 기준을 맞춰 주어야한다.

- 데이터 전처리의 중요성
    1. 주어진 데이터들의 기준이 다르다면 데이터를 바탕으로 test를 진행할때 원하는 결과가 나오지 않을 수 있다.
    2. 이를 위해서 샘플의 특성값을 일정한 기준으로 맞춰주는 작업을 데이터 전처리(data preprocessing)이라고 한다.
    3. 대표적인 전처리 방법으로는 표준점수를 이용하는것이다.

- 표준점수
    1. z score라고 표현하기도 한다.
    2. 표준점수: 각 특성값이 평균에서 표준편차의 몇배만큼 떨어져있는지를 나타낸다.
    3. 위와 같이 표준편차를 이용하여 전처리를 하게되면 실제 특성값의 크기와 상관없이 동일한 조건으로 비교가능하다.
    4. 표준점수를 이용한 전처리
        1. (특성값 - 평균) / 표준편차
        2. 각 행 열에 대해 하나씩 하기 힘들기 때문에 모든 행에 동시적용가능한 브로드캐스팅을 사용한다
        
        ```python
        train_scaled = (train_input - mean) / std
        ```
        
- 주어진 데이터를 정리할때 for문과 zip을 이용하는것 대신 np.column_stack, np.concatenate을 사용하여 간단히 배열을 만들 수 있다.

<img width="1510" height="564" alt="image" src="https://github.com/user-attachments/assets/a0a1e5e5-95d7-4823-b1be-03fc4d32cea9" />


- 간편하게 데이터세트와 훈련세트를 나누는 방법

```python
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, random_state=42)
```

- stratify를 추가해주면 데이터를 나눌때 target데이터의 클래스 비율대로 넣어주어 샘플링 편향을 없앨 수 있다.

```python
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify = fish_target, random_state=42)
print(test_target)
```

- k-최근접 이웃을 사용할때 찾고자 하는 샘플의 주변 샘플을 찾아주는 kneighbors메서드를 사용한다. 기본 이웃개수는 5이다.

```python
distances, indexes = kn.kneighbors([[25, 150]])
```
