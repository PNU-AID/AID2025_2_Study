# Chapter 02-1

1. **지도 학습** : **정답이 있음** → 알고리즘이 정답을 맞히는 것을 학습
    1. **입력** : 지도 학습에서의 데이터
    2. **타깃** : 지도 학습에서의 정답
    3. **훈련 데이터** : 입력 + 타깃
    4. **특성** : 입력으로 사용된 길이와 무게
  
       
       ![image](https://github.com/user-attachments/assets/f553cec2-ecfe-4cc2-9f37-c0b64f72e73a)


    6. 예시) **k-nn**
1. **비지도 학습** : **정답이 없음** → 타깃 없이 입력 데이터만 사용
2. **테스트 세트** : **평가**에 사용되는 데이터
3. **훈련 세트** : **훈련**(학습)에 사용되는 데이터
    1. **샘플** : 하나의 데이터
4. **샘플링 편향** : 훈련 세트와 테스트 세트의 샘플이 **골고루 섞여 있지 않아** 샘플링이 **한쪽으로 치우친** 상태 → 해결하기 위해 랜덤하게 샘플을 선택해 훈련 세트, 테스트 세트를 만들어야 함
5. 실습
    1. 인덱스, 슬라이싱 관련 내용 생략
    2. .shape() → numpy의 크기
    3. np.random.shuffle(index) → 인덱스 랜덤하게 섞기
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
        fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)] # 각 생선의 길이와 무게를 하나의 리스트로 담은 2차원 리스트
        # 도미 35마리, 빙어 14마리 -> 총 49개의 샘플
        fish_target = [1] * 35 + [0] * 14
        ```
        
        ```python
        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier()
        ```
        
        ```python
        train_input = fish_data[:35]  # 훈련 데이터 35개
        train_target = fish_target[:35] 
        
        test_input = fish_data[35:] # 테스트 데이터 14개
        test_target = fish_target[35:]
        ```
        
        ```python
        kn.fit(train_input, train_target)
        kn.score(test_input, test_target) # 정확도 0 -> 앞은 도미, 뒤는 빙어라서 샘플링 편향
        ```
        
        ```python
        import numpy as np
        ```
        
        ```python
        input_arr = np.array(fish_data)
        target_arr = np.array(fish_target)
        
        input_arr
        ```
        
        ```python
        input_arr.shape # 넘파이 배열 크기
        ```
        
        ```python
        np.random.seed(42) # 시드 설정 (값 안 변하게)
        index = np.arange(49) # 0~48까지 49개
        np.random.shuffle(index) # 인덱스를 랜덤하게 섞음
        
        index
        ```
        
        ```python
        train_input = input_arr[index[:35]] # 랜덤하게 배정한 인덱스의 처음 35개를 훈련 세트로
        train_target = target_arr[index[:35]]
        
        print(input_arr[13], train_input[0]) # 인덱스 첫 번째 값은 13 -> 처음 나오는 건 input_arr의 14번째 원소
        ```
        
        ```python
        test_input = input_arr[index[35:]] # 랜덤하게 배정한 인덱스의 나머지 14개를 테스트 세트로
        test_target = target_arr[index[35:]]
        ```
        
        ```python
        kn.fit(train_input, train_target)
        kn.score(test_input, test_target) # 정확도 1.0
        ```
        
        ```python
        kn.predict(test_input) # 예측 결과 -> array()로 감싸져 있음 (넘파이 배열임)
        ```
        
        ```python
        test_target # 실제 타깃
        ```
