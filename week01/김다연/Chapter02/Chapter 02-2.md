# Chapter 02-2

1. **스케일** : 두 특성의 값이 놓인 **범위**
2. **데이터 전처리** : 알고리즘의 특성값(샘플 간의 거리 등)을 **일정한 기준**으로 맞춰주는 작업
3. **표준점수**
    1. 각 특성값이 평균에서 **표준편차의 몇 배**만큼 떨어져 있는지 나타냄 → 스케일이 안 맞을 때 조정하는 방법 중 하나
    2. 계산법 : **(데이터 - 평균) ** 2**
4. **브로드캐스팅** : **모든 행**에 연산 적용

    ![image.png](https://github.com/user-attachments/assets/a552661b-412d-4f47-a1ce-bea8dbd341d7)
    
5. 실습
    1. column_stack() → 전달받은 리스트 일렬로 세우고 차례대로 나란히 연결 (연결할 리스트는 tuple로 전달)
    2. np.ones() → 1로 채운 배열 만듦, np.zeros() → 0으로 채운 배열 만듦
    3. np.concatenate() → 첫 번째 차원을 따라 배열 연결

        ![image.png](https://github.com/user-attachments/assets/e38d9906-f1ad-4748-83f0-48ba0428a8dd)
        
    4. stratify : 매개변수에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눔
    5. np.mean() → 평균 계산
    6. np.std() → 표준편차 계산
    - 코드
        
        ```python
        fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                        31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                        35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                        10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
        fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                        500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                        700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                        7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
        ```
        
        ```python
        import numpy as np
        ```
        
        ```python
        np.column_stack(([1, 2, 3], [4, 5, 6])) # 만들어진 배열 : (3, 2)
        ```
        
        ```python
        fish_data = np.column_stack((fish_length, fish_weight)) # fish_length와 fish_weight 합침
        
        fish_data[:5] # 잘 연결되었는지 확인
        ```
        
        ```python
        np.ones(5) # 1로 채운 배열 5칸
        ```
        
        ```python
        fish_target = np.concatenate((np.ones(35), np.zeros(14))) # 1로 채운 35개의 배열, 0으로 채운 14개의 배열 연결
        
        fish_target
        ```
        
        ```python
        from sklearn.model_selection import train_test_split
        ```
        
        ```python
        train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
        # fish_data는 input으로, fish_target은 target으로 각각 들어감
        
        print(train_input.shape, test_input.shape)
        ```
        
        ```python
        print(train_target.shape, test_target.shape) # 36개와 13개로 나뉨
        ```
        
        ```python
        test_target # 샘플링 편향 약간 나타남
        ```
        
        ```python
        train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
        
        test_target # 비율 맞춰짐
        ```
        
        ```python
        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier()
        kn.fit(train_input, train_target)
        kn.score(test_input, test_target)
        ```
        
        ```python
        kn.predict([[25, 150]]) # 0으로 예측
        ```
        
        ```python
        import matplotlib.pyplot as plt
        plt.scatter(train_input[:, 0], train_input[:, 1])
        plt.scatter(25, 150, marker='^') # marker : 모양 지정
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
        ```
        
        ```python
        distances, indexes = kn.kneighbors([[25, 150]])
        
        plt.scatter(train_input[:,0], train_input[:,1])
        plt.scatter(25, 150, marker='^')
        plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D') # 이웃 샘플 따로 구분해서 그림
        plt.xlabel('length')
        plt.ylabel('weight')
        ```
        
        ```python
        train_input[indexes]
        ```
        
        ```python
        train_target[indexes]
        ```
        
        ```python
        distances
        ```
        
        ```python
        plt.scatter(train_input[:,0], train_input[:,1])
        plt.scatter(25, 150, marker='^')
        plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
        plt.xlim((0, 1000))
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
        ```
        
        ```python
        mean = np.mean(train_input, axis=0)
        std = np.std(train_input, axis=0)
        ```
        
        ```python
        mean, std
        ```
        
        ```python
        train_scaled = (train_input - mean) / std
        ```
        
        ```python
        plt.scatter(train_scaled[:,0], train_scaled[:,1])
        plt.scatter(25, 150, marker='^') # 샘플도 동일하게 바꿔줘야 함
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
        ```
        
        ```python
        new = ([25, 150] - mean) / std
        plt.scatter(train_scaled[:,0], train_scaled[:,1])
        plt.scatter(new[0], new[1], marker='^')
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
        ```
        
        ```python
        kn.fit(train_scaled, train_target)
        ```
        
        ```python
        test_scaled = (test_input - mean) / std # 테스트 세트도 변환해줘야 함
        kn.score(test_scaled, test_target) # 정확도 1
        ```
        
        ```python
        kn.predict([new]) # 정확하게 분류
        ```
        
        ```python
        distances, indexes = kn.kneighbors([new])
        
        plt.scatter(train_scaled[:,0], train_scaled[:,1])
        plt.scatter(new[0], new[1], marker='^')
        plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D') # 이웃 샘플 따로 구분해서 그림
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
        ```
