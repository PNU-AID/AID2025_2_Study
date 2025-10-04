# 데이터 전처리
## 데이터 준비 및 가공
```python
# 데이터 준비
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
                
import numpy as np
fish_data = np.column_stack((fish_length, fish_weight)) # 2개 리스트(2열로) 연결
# np.concatenate - 하나의 행으로 연결, np.ones() - 배열을 1로 채움, np.zeros() - 배열을 0으로 채움
fish_target = np.concatenate((np.ones(35),np.zeros(14))) 
```
<img width="558" height="323" alt="image" src="https://github.com/user-attachments/assets/75bda37c-f4f0-405f-9a52-9f3ab6c9b227" />

## 훈련 세트와 테스트 세트 나누고 예측
```python
# 세트 분리(순수 무작위라 샘플링 편향 발생 가능)
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
# 세트 분리(클래스 비율 고려)
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
# 훈련 및 평가
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
#도미 데이터 넣어보기
print(kn.predict([[25, 150]])) # 잘못 예측
#산점도 확인
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
<img width="565" height="360" alt="image" src="https://github.com/user-attachments/assets/ab70e4b2-1c01-4bee-a1fa-dad822e23dcf" />
<img width="588" height="408" alt="image" src="https://github.com/user-attachments/assets/35e3e607-939c-4408-8159-1c91672e6afc" />
x, y축 scale이 달라서 문제 발생<br>

## x, y 축 scale 맞추기
```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
<img width="575" height="364" alt="image" src="https://github.com/user-attachments/assets/12ca3661-387d-46a5-94ba-3506a026fbc5" />
위 작업을 데이터 전처리라고 부릅니다.
## 표준점수
가장 널리 사용하는 전처리 방법 중 하나

### 계산법
```python
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std # 훈련 세트 보정
# 보정한 값으로 그린 산점도
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
<img width="586" height="359" alt="image" src="https://github.com/user-attachments/assets/72c75811-a8b5-4b03-90ef-704014fbb7e2" />
예측하려는 데이터를 보정하지않아 문제발생

### 예측하려는 데이터 보정
```python
# 예측하려는 데이터 보정 후 산점도
new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
<img width="555" height="377" alt="image" src="https://github.com/user-attachments/assets/36dc7114-19c4-4693-85a6-9882123ddbb9" />

### 테스트 세트 보정
```python
test_scaled = (test_input - mean) / std # 훈련세트 보정
kn.score(test_scaled, test_target) # 보정 후 평가
```

