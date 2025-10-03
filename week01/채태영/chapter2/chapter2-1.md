# Chapter 2-1

- Chapter 2-1 python code
    
    ```python
    fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                    31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                    35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                    10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
    fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                    500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                    700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                    7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
    fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
    fish_target = [1]*35 + [0]*14
    
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    
    print(fish_data[4])
    print(fish_data[0:5])
    print(fish_data[:5])
    print(fish_data[44:])
    --------------------------------------------------------------------------------------
    '''훈련세트와 테스트세트를 분리하였지만 데이터가 섞여있지 않아 
    훈련한 데이터를 바탕으로 테스트를 실행했을때 score가 0.0으로 나옴'''
    train_input = fish_data[:35]
    train_target = fish_target[:35]
    
    test_input = fish_data[35:]
    test_target = fish_target[35:]
    
    kn.fit(train_input, train_target)
    kn.score(test_input, test_target)
    --------------------------------------------------------------------------------------
    #위 문제를 해결하기 위해 numpy를 사용하여 데이터를 섞어줌.
    import numpy as np
    input_arr = np.array(fish_data)
    target_arr = np.array(fish_target)
    
    print(input_arr)
    print(input_arr.shape)
    
    np.random.seed(42)
    index = np.arange(49)
    np.random.shuffle(index) #index를 섞어주어 이를 기반으로 슬라이싱함.
    print(index)
    print(input_arr[[1, 3]])
    
    train_input = input_arr[index[:35]]
    train_target = target_arr[index[:35]]
    print(input_arr[13], train_input[0])
    
    test_input = input_arr[index[35:]]
    test_target = target_arr[index[35:]]
    
    import matplotlib.pyplot as plt
    
    plt.scatter(train_input[:, 0], train_input[:, 1])
    plt.scatter(test_input[:, 0], test_input[:, 1])
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
    kn.fit(train_input, train_target)
    kn.score(test_input, test_target)
    kn.predict(test_input)
    test_target
    
    ```
    

머신러닝은 크게 지도학습, 비지도학습으로 나눌 수 있다.

<img width="800" height="416" alt="image" src="https://github.com/user-attachments/assets/08ef1b96-d1fc-48e6-bb64-fc5d6280c910" />


- 지도학습
    1. 데이터와 정답을 각각 입력, 타킷이라고 부르며 이 둘을 합쳐 훈련데이터라고 한다.
    2. 특성(feature) : 입력으로 사용되는 특징
    3. 타깃을 바탕으로 알고리즘이 정답을 맞히는걸 학습하는것이 지도학습
    
- 훈련세트와 테스트세트
    1. 준비된 데이터중 일부는 평가에 사용하는 테스트세트, 일부는 훈련에 사용되는 훈련세트로 나눈다.
    2. 훈련상황을 평가할때 사용하는 테스트세트는 훈련할때 사용되지 않은 데이터를 사용해야한다.
    3. 훈련세트와 테스트세트에 샘플이 골고루 섞여있지 않으면 샘플링편향이 생긴다.
        1. 아래의코드와 같이 훈련세트에는 도미만 있고 테스트세트에는 빙어만 있게된 경우, kn.score를 하면 0.0이 뜬다.
        
        ```python
        train_input = fish_data[:35]
        train_target = fish_target[:35]
        
        test_input = fish_data[35:]
        test_target = fish_target[35:]
        ```
