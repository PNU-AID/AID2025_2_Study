# chapter1 나의 첫 머신러닝
## 1-1 인공지능과 머신러닝, 딥러닝
인공지능 - 사람처럼 학습하고 추론할 수 있는 지능을 가진 컴퓨터 시스템을 만드는 기술<br>
머신러닝 - 규칙을 일일이 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야<br>
딥러닝 - 머신러닝 알고리즘 중에 인공 신경망을 기반으로 한 방법들을 통칭하여 부르는 말<br>
### 사이킷런
<img width="257" height="190" alt="image" src="https://github.com/user-attachments/assets/12c6ebba-4baa-4009-932a-97453d4d03f9" /><br>
대표적인 머신러닝 라이브러리<br>

## 1-2 코랩과 주피터 노트북
### 구글 코랩<br>
<img width="385" height="250" alt="image" src="https://github.com/user-attachments/assets/84dae46d-664b-4ab8-9c07-af386a216eaa" /><br>
웹 브라우저에서 무료로 파이썬 프로그램을 테스트하고 저장할 수 있는 서비스(=클라우드 기반 주피터 노트북 개발 환경)
## 1-3 마켓과 머신러닝
### 생선 분류 문제
#### 1. 도미 데이터 준비
```python
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
```
위와 같은 length, weight를 특성이라 한다.<br>
#### 2. 빙어 데이터 준비
```python
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```
- 산점도
```python
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
<img width="557" height="360" alt="image" src="https://github.com/user-attachments/assets/e1a9a7b7-c45d-4515-b725-26d5610aee4f" />
주황(빙어), 파랑(도미)<br>

#### 3. 구분 by k-최근접 이웃 알고리즘

k-최근접 이웃 알고리즘 - 주위 의 데이터 중 다수를 차지하는 것을 정답 채택(기본값 5, n_neighbors=n으로 변경 가능)

```python
# 두 리스트 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
# 2차원 리스트 만들기
fish_data = [[l,w] for l, w in zip(length, weight)]
# 도미를 1, 빙어를 0으로 표현
fish_target = [1] * 35 + [0] * 14
# k-최근접 알고리즘 임포트
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target) # 훈련
kn.score(fish_data, fish_target) # 정확도(0~1)
kn.predict([[30, 600]]) # 정답 예측
```


