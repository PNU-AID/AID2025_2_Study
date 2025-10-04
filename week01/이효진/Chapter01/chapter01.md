# 01

---

인공지능 : 사람처럼 학습하고 추론할 수 있는 지능을 가진 시스템을 만드는 기술임.

딥러닝 : 인공 신경망을 기반으로 한 분야임.

**머신러닝 : 규칙을 프로그래밍하지 않고 자동으로 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야임.**

- 분류 : 여러 개의 종류 중 하나를 구별해 내는 문제. 2개 클래스 중 하나 고르는건 이진 분류라고 함.

**특성 : 데이터의 특징**

```cpp
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
# 도미 데이터 준비
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 도미의 산점도

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
# 빙어 데이터 준비
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 도미와 빙어의 산점도
```

**k-최근접 이웃 알고리즘 : 답을 구하려는 데이터의 주변에 있는 데이터를 확인하고 그 중 다수인 것을 답으로 사용함.**

정확도 = 정확히 맞힌 개수 / 전체 데이터 개수 

```cpp
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
# 특성끼리 리스트 합침. 
fish_data = [[l, w] for l, w in zip(length, weight)]
# 사이킷런을 사용하려면 리스트를 세로방향 2차원 리스트로 만들어야함.
fish_target = [1]*35 + [0]*14
# 정답 데이터 (찾으려는 대상을 1로 , 그 외에는 0으로 놓음.)
from sklearn.neighbors import KNeighborsClassifier # k-최근접 이웃 클래스
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target) # 훈련시킴.
kn.score(fish_data, fish_target) # 모델 평가. 0~1 사이 값 반환, 클수록 정확히 맞춘것.
kn.predict([[30, 600]])
#print(kn._fit_X)
#print(kn._y)
kn49 = KNeighborsClassifier(n_neighbors=49) 
# 가까운 49개의 데이터를 사용하면 모든 생선을 사용하여 예측, 즉 어떤 데이터를 넣어도 도미로 예측해버림.
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
```

훈련 : 머신러닝 알고리즘이 데이터에서 규칙을 찾는 과정. 사이킷런에서 fit() 의 역할임.

모델 : 머신러닝 알고리즘을 구현한 프로그램을 말함.