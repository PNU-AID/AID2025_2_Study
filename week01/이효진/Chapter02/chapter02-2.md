# 02-2 | 데이터 전처리

---

```cpp
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
import numpy as np
# 직접 리스트 안 만들고 넘파이로 간편하게 만들 수 있음
fish_data = np.column_stack((fish_length,fish_weight)) # 세로로 두 리스트 연결
fish_target = np.concatenate((np.ones(35), np.zeros(14))) # 가로로 두 리스트 연결

from sklearn.model_selection import train_test_split
# 인덱스 직접 섞지않고 train_test_split 함수 이용해 train set, test set으로 나눔.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
# 무작위로 하면 샘플이 골고루 섞이지 않을 수 있음 -> stratify 매개변수에 타겟데이터 전달해 클래스 비율에 맞게 데이터 나눔.
```

```cpp
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
# k-최근접 이웃 모델 훈련. 근데 predict하면 예측 이상하게함.
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 산점도를 보면 도미에 가까워 보이는데도 모델은 빙어 데이터에 가깝다고 판단함
distances, indexes = kn.kneighbors([[25,150]])
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150,marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 근처 5개 이웃 샘플 구분해서 보면 4개가 빙어, 1개가 도미인 것을 알수있음.
```

그래프에 나타나는 거리와 실제 샘플 거리 비율이 다름

→x축,y축 값의 범위가 다르기 때문임. 이를 **스케일**이 다르다고 함.

데이터 전처리 : 머신러닝 모델에 train data 넣기 전에 가공하는 단계

1. **표준점수 (z점수) : train set의 스케일을 바꾸는 대표적인 방법임. 특성의 평균을 뺴고 표준편차로 나누면 됨. 이때  test set은 반드시 train set의 평균과 표준편차로 바꿔야 함!!**

브로드캐스팅 : 넘파이 기능으로, 사칙연산을 모든 행, 열에 적용해줌.

```cpp
mean = np.mean(train_input, axis=0) # 평균 계산
std = np.std(train_input, axis=0) # 표준편차 계산
train_scaled = (train_input - mean) / std
# train set 표준점수 변환
new = ([25, 150] - mean) / std
kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
# 반드시 test set 을 train set의 평균과 표준편차로 스케일 변환
kn.score(test_scaled, test_target)
# 1.0 으로 잘 예측함.
```