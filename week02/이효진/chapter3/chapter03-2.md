# 03-2 | 선형 회귀

---

k-최근접 이웃 회귀는 가까이 있는 샘플을 찾아 타겟을 평균으로 구하기 때문에 ,새로운 샘플이 train set의 범위를 벗어나면 이상하게 예측할 수 있음.

→ 다른 알고리즘, 선형 회귀 사용

**선형 회귀 : 특성과 타겟 사이의 관계를 가장 잘 나타내는 선형 방정식 찾기. 특성 하나면 최적의 직선 찾기.**

$y=ax+b$  에서 a→coef   , b→intercept . a와 b는 모델 파라미터라고 불림.

coef, 즉 기울기는 계수(coefficient) 또는 가중치(weight)라고 불림.

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input,train_target)
print(lr.predict([[50]]))
print(lr.coef_,lr.intercept_)
# [1241.83860323]
# [39.01714496] -709.0186449535477
```

농어의 길이 15~50까지 직선으로 그려보면 다음과 같음.

```python
plt.scatter(train_input,train_target)
plt.plot([15,50],[15*lr.coef_+lr.intercept_,50*lr.coef_+lr.intercept_])
plt.scatter(50,1241.8,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![혼공머신선형회귀.png](%ED%98%BC%EA%B3%B5%EB%A8%B8%EC%8B%A0%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80.png)

**다항 회귀 : 최적의 곡선 찾기. 다항식을 사용한 선형 회귀임.**

$y=ax^2+bx+c$  에서 a,b→ coef, c→intercept

```python
# 다항 회귀
train_poly = np.column_stack((train_input**2,train_input))
test_poly = np.column_stack((test_input**2,test_input))
print(train_poly.shape,test_poly.shape)

lr = LinearRegression()
lr.fit(train_poly,train_target)
print(lr.predict([[50**2,50]]))
print(lr.coef_,lr.intercept_)
# [1573.98423528]
# [  1.01433211 -21.55792498] 116.0502107827827
```

![혼공머신다항회귀.png](%ED%98%BC%EA%B3%B5%EB%A8%B8%EC%8B%A0%EB%8B%A4%ED%95%AD%ED%9A%8C%EA%B7%80.png)