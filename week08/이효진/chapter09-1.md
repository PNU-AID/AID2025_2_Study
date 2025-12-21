# 09-1 | 순차 데이터와 순환 신경망

### 순차 데이터

: 텍스트나 시계열 데이터와 같이 순서에 의미가 있는 데이터

**feedforward neural network** : 입력 데이터의 흐름이 앞으로만 전달되는 신경망

<img width="393" height="161" alt="image" src="https://github.com/user-attachments/assets/a2c561fd-d472-403e-8825-e38c6b87df87" />


순차 데이터를 다루기 위해선 이전 데이터를 기억하는 기능이 필요한데, 이때까지 배운 합성곱 신경망은 피드포워드 신경망이라 기억 장치가 없다.

### 순환 신경망 ( RNN )

: 이전 데이터가 신경망 층에 순회되는 신경망

<img width="207" height="89" alt="image 1" src="https://github.com/user-attachments/assets/ba04fb10-3e99-4e48-b932-64084e83cefa" />


<img width="215" height="110" alt="image 2" src="https://github.com/user-attachments/assets/1d552244-8d01-4c2a-a49b-8f3a9d2a7ba0" />


<img width="226" height="110" alt="image 3" src="https://github.com/user-attachments/assets/ebba33ab-90cb-4b16-8626-7abfcc4d7628" />


**타임 스텝 (timestep)** : 시퀀스에서 샘플을 처리하는 한 단계 

- 시퀀스가 있다 → `[x₁, x₂, x₃, …, x_T]`
- RNN은 이걸 시간 방향으로 한 개씩 넣는다
- 이때 x₁, x₂, x₃ …를 각각 처리하는 “한 단계”가 타임스텝

**셀 (cell) :** 층

**은닉 상태 (hidden state)** : 셀의 출력

<img width="220" height="132" alt="image 4" src="https://github.com/user-attachments/assets/0406c813-7e92-4f45-ab2e-2cd12eee0878" />


그림과 같이 은닉 상태를 다음 타임스텝에 재사용한다.

**하이퍼볼릭 탄젠트 (hypernbolic tangent)** : 은닉층의 활성화 함수로 주로 사용되며, -1에서 1사이의 범위를 가진다.

<img width="182" height="203" alt="image 5" src="https://github.com/user-attachments/assets/85acefe5-b15f-49e1-bc8f-3776873bb6be" />


순환 신경망은 이전 타임스텝의 은닉 상태에 곱해지는 가중치( $w_h$ )가 추가로 존재한다.

### 셀의 가중치와 입출력

순환층에 입력되는 특성의 개수가 4개, 뉴런이 3개일 때

가중치 $w_x$의 크기 : 4 * 3 = 12   

가중치 $w_h$의 크기 : 3 * 3 = 9 ,  이전 타임스텝의 은닉 상태는 다음 타임스텝의 뉴런에 연결됨

각 뉴런 당 절편 : 1*3=3

순환층 총 모델 파라미터 개수 12+9+3=24 

순환층의 입력은 시퀀스 길이와 단어 표현으로 이루어진 2차원 배열구조를 가진 샘플이다.

순환층을 여러개 쌓았을 때, 마지막 셀을 제외한 다른 모든 셀은 모든 타임스텝의 은닉 상태를 출력한다.

순환층의 출력층은 마지막 셀의 출력이 항상 1차원 벡터이다.
