# 03-1 | k-최근접 이웃 회귀

---

분류 : 샘플을 클래스 중 하나로 분류하는 문제

**회귀 : 임의의 어떤 숫자를 예측하는 문제**

k-최근접 이웃 분류 : 예측하려는 샘플에 가장 가까운 샘플 k개를 선택, 클래스 확인 후 최다 클래스로 예측

**k-최근접 이웃 회귀 : 예측하려는 샘플에 가장 가까운 샘플 k개를 선택, k개의 샘플 평균 값으로 예측**

```python
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = **train_test_split**(perch_length,perch_weight,random_state=42)
# 훈련 세트와 테스트 세트로 나눔.
train_input = train_input.**reshape(-1,1)**
test_input = test_input.**reshape(-1,1)**
# 사이킷런에 사용할 세트들은 2차원 배열이여야 하므로 reshape 이용해 2차원 배열로 바꿈. 
# 특성 한 개 (길이) 이므로 열이 하나인 배열 만들어야 함.
# reshape(-1,1) : 첫번째 크기를 나머지 원소 개수에 맞게 설정하고 두번째 크기를 1로 함.
print(train_input.shape, test_input.shape)
# 결과 : (42,1) (14,1)
```

**결정계수(  $R^2=1-\frac{(타겟-예측)^2의\space합}{(타겟-평균)^2의\space합}$)  :  회귀 문제의 성능 측정 도구. 타겟의 평균 정도를 예측하는 수준이면 0에 가까워지고, 타겟에 아주 가까워지면 1에 가까운 값이 됨.**

mean_absolute_error : 타겟과 예측의 절댓값 오차의 평균

```python
# 결정계수
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))

# mean_absolute_error
from sklearn.metrics import mean_absolute_error
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```

**과대적합 (overfitting) : train set에 너무 잘 맞춰진 모델**

**과소적합 (underfitting) : train set에 적절히 훈련되지 않음 / train set, test set 모두 $R^2$ 점수가 낮은 경우.**

→ 과소적합 어떻게 해결하나요???

—> 이웃의 개수 k를 줄여 train set 에 잘 맞게 만들면 됨.

```python
# k 개수 줄이기
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))
# 이제 train set 점수가 높아짐. 과소적합 해결
```

→ 과대적합일땐?

—> 이웃의 개수 k를 늘려 일반적인 패턴에 잘 맞게 만들면 됨.

결론은 test set, train set의 점수차를 적게, train set 점수 > test set점수  가 되도록 만들면 된다!