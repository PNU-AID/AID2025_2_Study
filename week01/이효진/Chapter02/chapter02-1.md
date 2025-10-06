# 02-1 | 훈련 세트와 테스트 세트

---

**지도 학습 :  입력과 타겟으로 모델을 훈련한 다음 새로운 데이터를 예측하는 것. 데이터를 입력, 정답을 타겟이라하고 이 둘을 합쳐 훈련 데이터라고 함.**

비지도 학습 : 타겟 없이 입력 데이터만 사용해 어떠한 특징을 찾음.

**train set : 모델을 훈련할 때 사용하는 데이터**

**test set : 전체 데이터의 20~30%를 모델을 테스트 하는 데이터로 사용함.**

머신러닝 알고리즘의 성능을 제대로 평가하기 위해선 train data와 test data 가 달라야함. 

일반적으로 이미 준비된 데이터에서 8 : 2 정도의 비율으로 train set, test set으로 나눠 평가함.

```cpp
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
# 2차원 리스트 만들기. 49개의 샘플이 있고 특성은 길이와 무게이다.
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
print(fish_data[4])

train_input = fish_data[:35]
train_target = fish_target[:35]
# train set으로 0~34 인덱스 사용
test_input = fish_data[35:]
test_target = fish_target[35:]
# test set으로 35~마지막 인덱스까지 사용

kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
# 결과 : 0.0 나옴.
```

→train set에 only 도미, test set에 only 빙어만 들어있기 때문에 정답 못 맞춤.

**샘플링 편향 : train set과 test set에 샘플이 골고루 섞여 있지 않고 한쪽으로 치우친 경우.**

이를 해결하기 위해서 샘플을 골고루 train set, test set에 분배해줘야 한다. 파이썬의 넘파이를 사용한다.

```cpp
import numpy as np
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
np.random.seed(42)
index = np.arange(49)
# arrange이용해 0부터 48까지 1씩 증가하는 인덱스 만듦
np.random.shuffle(index)
# 인덱스를 무작위로 섞음.
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
# 인덱스 배열의 35개의 샘플을 train set으로 만듦
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
# 인덱스 배열의 나머지 14개 샘플을 test set으로 만듦
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 도미와 빙어가 잘 섞였는지 산점도 그리기. 결과보면 잘 섞여 있음.
# k-최근접 이웃 모델로 훈련
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
kn.predict(test_input)
test_target
# 훈련이 잘 되었는지 확인.
```