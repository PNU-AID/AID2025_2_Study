# 09-2 | 순환 신경망으로 IMDM 리뷰 분류하기

### 순환 신경망 만들기

토큰을 정수로 변환한 데이터를 신경망에 주입하면 큰 정수가 큰 활성화 출력을 만든다

→ 큰 토큰일수록 중요시하게 된다. (ex. ‘20이 10보다 두 배 중요하다 ‘라고 착각함)

원-핫 인코딩으로 해결 ( to_categorical() )

```cpp
train_oh = keras.utils.to_categorical(train_seq)
```

### 단어 인베딩을 사용하기

원-핫 인코딩은 입력 데이터가 너무 커진다는 단점이 있음. 

이를 해결하기 위해 **각 단어를 고정된 크기의 실수 벡터로 바꿔주는 단어 임베딩**을 사용함.

<img width="330" height="60" alt="image" src="https://github.com/user-attachments/assets/6dec9fbf-32ba-459c-a28e-2675ceba0348" />


훨씬 작은 크기로 단어를 잘 표현할 수 있게됨→메모리 절약, 더 많은 단어 사용가능
