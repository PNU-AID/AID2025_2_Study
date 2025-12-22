# 08-1 | 합성곱 신경망의 구성 요소

### 합성곱

밀집층에서는 모든 입력에 가중치를 곱함.

<img width="308" height="178" alt="image" src="https://github.com/user-attachments/assets/fa2fba89-23de-449c-bd0a-d8d9172e27c1" />


**합성곱은 일부 입력에 가중치를 곱함.**

<img width="244" height="193" alt="image 1" src="https://github.com/user-attachments/assets/557ce796-6c00-4e2d-9713-5c492baac48a" />


합성곱 신경망 (CNN) 에서는 뉴런을 필터(=커널)이라고 부름. 

합성곱 출력을 특성 맵(feature map)이라고 함.

2차원 구조를 그대로 사용 가능→이미지 처리에서 뛰어난 성능을 가짐

### 케라스 합성곱 층

```python
keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu')
# 필터의 개수 / 필터에 사용할 커널의 크기 / 활성화 함수
```

합성곱 신경망 : 1개 이상의 합성곱 층을 쓴 인공 신경망

패딩과 스트라이드

1. 패딩 : 입력 배열의 주위를 가상의 원소로 채우는 것
    
    커널의 크기를 유지하면서 입력과 동일한 크기의 출력을 만들려면 임의로 실제 입력 크기보다 큰 입력에 합성곱을 해야함. 이 과정에서 패딩 사용
    
    ex. ) 입력-(4,4) / 커널-(3,3) / 출력-(2,2) → 입력-(6,6) / 커널-(3,3) / 출력-(4,4)
    
    <img width="446" height="487" alt="image 2" src="https://github.com/user-attachments/assets/1a15e5dc-a055-4097-ac42-a7e686527bad" />

    
    세임 패딩 : 입력과 특성 맵의 크기를 동일하게 만들기 위해 입력 주위를 0으로 패딩
    
    밸리드 패딩 : 패딩 없이 입력 배열에서만 합성곱을 하여 특성 맵을 만드는 것
    
    Q. 패딩 사용하는 이유?
    
    A. 이미지 주변에 있는 정보를 잃어버리지 않도록 함.
    
    ```python
    keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu', 
    																						padding='same')
    																						 # 세임 패딩
    ```
    
2. 스트라이드 : 커널 이동의 크기
    
    ```python
    keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu',
    												padding='same', strides=1)
    												               # 기본값. 한칸씩 이동
    ```
    

풀링 : 특성 맵의 가로세로 크기를 줄이는 역할.

```python
# 최대 풀링
keras.layers.MaxPooling2D(2) # 플링의 크기
= keras.layers.MaxPooling2D(2,strides=2,padding='valid')

# 평균 풀링
AveragePooling2D
```

최대 풀링 : 이동한 각 영역에서 가장 큰 값을 저장.

<img width="325" height="194" alt="image 3" src="https://github.com/user-attachments/assets/874599ad-9162-4752-85af-c3a8e440edba" />


평균 풀링 : 평균값 계산

### 합성곱 신경망의 전체 구조

<img width="475" height="187" alt="image 4" src="https://github.com/user-attachments/assets/4d990b51-1f51-492c-8e9b-fafb5ab58709" />


1. 입력 (4,4) 
2. 합성곱 층
    
    → 필터 3개 , (3,3) 크기 , 특성맵 크기 (4,4,3) 
    
3. 풀링 층
    
    → (4,4,3) 을 (2,2,3) 으로 가로세로 크기 절반으로 줄임
    
4. 밀집층 
    
    뉴런 3개인 출력층에 전달하기 위해 3차원 배열 (2,2,3) 을 1차원(12)으로 펼침.
    

컬러 이미지를 사용한 합성곱 : 3차원 배열일때 합성곱 수행해야함.

<img width="295" height="154" alt="image 5" src="https://github.com/user-attachments/assets/b24b3b07-7555-46ff-99a7-f2ac15e4e2e6" />


= 합성곱층-풀링층-합성곱층 일 경우

<img width="439" height="245" alt="image 6" src="https://github.com/user-attachments/assets/f20a2a8a-8c23-4ac6-959e-d682da0caa0b" />


합성곱 신경망에서 필터는 이미지에 있는 어떠한 특징을 찾는다고 할수있다. 

너비와 높이는 점점 줄어들고(어떤 특징이 이미지의 어느 위치에 있더라도 쉽게 감지 가능하도록)

깊이는 점점 깊어진다(처음엔 간단한 특징을 찾고 층이 깊어질수록 다양하고 구체적인 특징을 감지하도록)
