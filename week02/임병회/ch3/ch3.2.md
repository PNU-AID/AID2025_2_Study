# 선형회귀
## k-최근접 이웃의 한계
```python
# 데이터 준비
import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
# 세트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
# 세트 2차원 배열로
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
# 최근접 이웃 개수 3인 모델 훈련
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)
print(knr.predict([[50]])) # 길이 50cm 농어 무게 예측
# 실제 농어 무게보다 많이 낮게 산출, 기존 데이터들이 무게가 많이 낮아서 발생
```

## 선형회귀
대표적인 회귀 알고리즘<br>
<img width="293" height="178" alt="image" src="https://github.com/user-attachments/assets/496c75a5-3b4b-4b38-a9c1-f9d56b37ccc3" />

```python
# 선형 회귀 모델 훈련 및 예측
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[50]]))
# 전보다 무게 높게 산출
```
<img width="496" height="324" alt="image" src="https://github.com/user-attachments/assets/3d6efaad-8d53-409d-a0d0-6127a949e6a9" />

### a, b 구하기
```python
# 기울기, 절편 구하기
print(lr.coef_, lr.intercept_)
# 산점도 
plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
plt.scatter(50, 1241.8, marker='^')
plt.show()
```
<img width="552" height="359" alt="image" src="https://github.com/user-attachments/assets/ac743e4b-a092-4d14-8ea0-66347c517279" />

```python
# 평가
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))
# 위 그래프 왼쪽 아래로 인해 과소적합
```

## 다항회귀
위와 같이 과소적합되는 문제 해소를 위해 최적 직선 아닌 최적 곡선을 찾는 것<br>
<img width="514" height="334" alt="image" src="https://github.com/user-attachments/assets/8a15d314-e75b-4581-99cf-ef15243a6c56" />
```python
# 제곱한 항 추가
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
# 훈련 및 예측
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
# 계수, 절편 확인
print(lr.coef_, lr.intercept_)
# 산점도
point=np.arange(15, 50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
plt.scatter([50], [1574], marker='^')
plt.show()
```
<img width="596" height="379" alt="image" src="https://github.com/user-attachments/assets/70fb9826-d6b5-43a9-85c1-8acf0ba93fb5" />

```python
# 평가
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# 직선 때보단 나아졌지만 과소적합
```

