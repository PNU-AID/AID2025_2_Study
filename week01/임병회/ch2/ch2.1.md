# 훈련 세트와 테스트 세트
## 지도학습과 비지도 학습
<img width="365" height="290" alt="image" src="https://github.com/user-attachments/assets/b6fe5fdd-5104-4753-b32e-47addf0fbad0" /><br>
<지도학습에 사용되는 것들><br>
훈련세트 - 훈련에 사용되는 데이터<br>
테스트세트 - 평가에 사용되는 데이터<br>
### 샘플링이 편향된 경우
```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

#2차원 리스트 만들기
fish_data = [[l,w] for l,w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

# KNeighbborclassifier 임포트
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

# 훈련세트 0-34 인덱스 사용
train_input = fish_data[:35]
train_target = fish_target[:35]
# 테스트 세트 35- 인덱스 사용
test_input = fish_data[35:]
test_target = fish_target[35:]

# 샘플이 안섞여 정확도 0 산출
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```
### 넘파이를 통해 샘플링 편향 해결
넘파이 - 파이썬의 대표적인 배열 라이브러리, 고차원 리스트 표현 시 사용
```python
# numpy 호출
import numpy as np

# numpy 배열로 변경
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
#input_arr, targer_arr 같은 위치에 있어야함

#인덱스 랜덤하게 섞기
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

#섞은 데이터 훈련, 테스트 세트 분리
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

#산점도를 통해 섞임 여부 확인
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
<img width="587" height="383" alt="image" src="https://github.com/user-attachments/assets/199d91d6-b23d-47d7-815a-beefceeebdd5" />
파랑(훈련세트), 주황(테스트 세트)

```python
#훈련 및 테스트
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
