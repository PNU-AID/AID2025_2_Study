# 특성 공학과 규제

## 다중회귀
여러 개의 특성을 사용한 선형회귀<br>
특성공학 - 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업<br>

```python
# 판다스에서 다차원 배열 데이터 호출
import pandas as pd
df = pd.read_csv('http://bit.ly/perch_csv')
perch_full=df.to_numpy()
# 타겟 데이터 호출
import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
# 세트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
# 새로운 특성 만들기
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
poly.get_feature_names_out() # 특성이 어떻게 만들어졌는 지 확인
test_poly = poly.transform(test_input) # 테스트 세트 변환
# 훈련
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target)) # 훈련세트 점수 확인
print(lr.score(test_poly, test_target)) # 테스트 세트 점수 확인
# 과소적합 해결

# if 특성을 추가한다면?
# 특성을 5제곱까지 추가
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape) # 특성 55개가 됨
# 훈련 및 평가
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target)) # 좋은 점수
print(lr.score(test_poly, test_target)) # 훈련세트에 과대적합되어 음수가 나옴(문제 있음)
```

## 규제
머신러닝 모델이 훈련 세트에 과대적합되는 걸 방해하는 것

```python
# 규제
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

### 릿지 회귀

```python
# 릿지회귀
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) # 릿지 전보다 낮아짐
print(ridge.score(test_scaled, test_target)) # 음수인 것 해결
# alpha값으로 규제 강도 조절
# 리스트 생성
import matplotlib.pyplot as plt
train_score = []
test_score = []
# 적절 alpha 찾기
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
# x축 log로 동일간격 맞춘 후 그래프 산출
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.show()
```

<img width="598" height="381" alt="image" src="https://github.com/user-attachments/assets/7bf0a0fa-5dbe-4f9f-9555-88fd82354ec9" />

가장 가까운 alpha값인 10<sup>-1</sup>로 결정

```python
# alpha 0.1로 훈련 및 평가
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
# 적절한 균형 맞아짐
```

### 라쏘 회귀

```python
# 라쏘 불러오고 훈련 및 평가
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
# 나쁘지 않은 결과
# 적절 alpha 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
# 그래프 산출
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.show()
```

<img width="580" height="379" alt="image" src="https://github.com/user-attachments/assets/b837fc36-0e91-4a78-90e9-772a4c3c2992" />
가장 가까운 10이 적절 alpha

```python
# alpha 10으로 훈련 및 평가
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
# 과대 적합 잘 억제
```
