# 03-3 | 특성 공학과 규제

---

**다중 회귀 : 여러 개의 특성을 사용한 선형 회귀**

지금까지 농어의 길이만을 사용해 예측해 왔다면, 이젠 두께와 높이까지 함께 사용함.

**특성 공학 : 기존의 특성을 사용해 새로운 특성을 만드는 작업**

ex) 각 특성을 곱해서 또 다른 특성을 만듦. (농어길이 * 농어 높이→새로운 특성)

사이킷런에 특성을 만들 수 있는 PolynomialFeatures 라는 클래스가 있음.  

각 특성을 제곱한 항과 특성끼리 서로 곱한 항을 추가함.

```python
from sklearn.preprocessing import PolynomialFeatures
# 예시
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2,3]]) # 2와 3이라는 두 특성을 이용해
print(poly.transform([[2,3]])) # 새로운 특성 만듦.
# 결과 :  [[2. 3. 4. 6. 9.]]

# 농어에 적용
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input) # train set 변환
print(train_poly.shape)
# 결과 : (42, 9) , 9개의 특성이 만들어짐.

poly.get_feature_names_out() # 특성이 어떤 조합으로 만들어졌는지 확인 가능.
# array(['length', ' height', ' width', 'length^2', 'length  height',
#      'length  width', ' height^2', ' height  width', ' width^2'],
#     dtype=object)
test_poly = poly.transform(test_input) # test set 변환
```

다중 회귀 모델 훈련은 선형 회귀 모델을 훈련하는 것과 같음.

degree 매개변수로 최대 차수를 지정해 특성을 더 추가할 수 있음.

```python
# 특성 추가. degree=5 로 5제곱 항 넣음.
poly = PolynomialFeatures(degree=5, include_bias=False) 
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
# 결과 : (42, 55) , 특성 55개

# 재훈련
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# 결과 :  0.9999999999996433
# 결과 : -144.40579436844948
# 특성 개수 너무 많이 늘려서 train set에 과대 적합됨. test set의 점수 매우 낮아짐.
```

**규제 : 머신러닝 모델이 train set에 과대적합되지 않도록 만드는 것**

→선형 모델의 경우 특성에 곱해지는 계수(기울기)의 크기를 작게 만드는 일

보편적인 패턴을 학습하도록!

규제 적용 전 특성의 정규화 필요함.  StandardScalar클래스 사용

```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

**릿지 (ridge) : 계수를 제곱한 값을 기준으로 규제 적용**

**라쏘 (lasso) : 계수의 절댓값을 기준으로 규제 적용**

alpha 매개변수로 규제 강도 조절 가능함.  

하이퍼파라미터로 머신러닝 알고리즘이 학습하지 않아 사람이 사전에 지정해야함.

알파 값이 커지면 규제 강하게 작용, 계수 값 줄이고 과소적합되도록 함. (test set에 맞게)

알파 값이 작아지면 규제 약하게 작용, 과대적합되도록 함. (train set에 맞게)

→적절한 알파 값을 찾는 방법?

—> 알파 값에 대한 $R^2$값의 그래프를 그려 train set과 test set의 점수가 가장 가까운 지점을 찾음.

```python
# 릿지 회귀 
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))
# alpha 매개변수로 규제강도 조절 가능 
import matplotlib.pyplot as plt
train_score = []
test_score = []
alpha_list = [0.001,0.01,0.1,1,10,100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled,train_target)
    train_score.append(ridge.score(train_scaled,train_target))
    test_score.append(ridge.score(test_scaled,test_target))
plt.plot(alpha_list,train_score)
plt.plot(alpha_list,test_score)
plt.xscale('log')
plt.ylabel('alpha')
plt.ylabel('R^2')
plt.show()
```

![왼쪽은 train set점수 아주 높고 test set점수 아주 낮은 과대적합, 오른쪽으로 갈수록 두 점수 모두 낮아지는 과소적합](%ED%98%BC%EA%B3%B5%EB%A8%B8%EC%8B%A0%EB%A6%BF%EC%A7%80.png)

왼쪽은 train set점수 아주 높고 test set점수 아주 낮은 과대적합, 오른쪽으로 갈수록 두 점수 모두 낮아지는 과소적합

```python
# 두 그래프가 가장 가깝고 test set 점수가 가장 높은 10^-1=0.1 로 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))
# 0.9903815817570367
# 0.9827976465386928
```

```python
# 라쏘 회귀
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))
print(lasso.score(test_scaled,test_target))
# 규제 강도 조절
train_score = []
test_score = []
alpha_list = [0.001,0.01,0.1,1,10,100]
for alpha in alpha_list:
  lasso = Lasso(alpha=alpha)
  lasso.fit(train_scaled,train_target)
  train_score.append(lasso.score(train_scaled,train_target))
  test_score.append(lasso.score(test_scaled,test_target))
plt.plot(alpha_list,train_score)
plt.plot(alpha_list,test_score)
plt.xscale('log')
plt.ylabel('alpha')
plt.ylabel('R^2')
plt.show()
```

![동일하게 왼쪽은 과대적합, 오른쪽은 과소적합.](%ED%98%BC%EA%B3%B5%EB%A8%B8%EC%8B%A0%EB%9D%BC%EC%8F%98.png)

동일하게 왼쪽은 과대적합, 오른쪽은 과소적합.

```python
# 10^1=10 에서 모델 훈련
lasso = Lasso(alpha=10)
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))
print(lasso.score(test_scaled,test_target))
# 0.9888067471131867
# 0.9824470598706695
```