# 09-3 | LSTM과 GRU 셀

기본 순환층은 시퀀스가 길수록 순환되는 은닉 상태에 담긴 정보가 희석되기 때문에, 긴 시퀀스를 학습하기 어렵다.

### LSTM 구조

Long Short-Term Memory 의 약자, 단기 기억을 오래 기억하기 위해 고안됨.

LSTM에는 순환되는 상태가 2개

1. 은닉 상태 h
2. 셀 상태 c : 다음 층으로는 전달하지 않고 셀 내부에서만 순환
    1. 삭제 게이트 : 셀 상태에 있는 정보를 제거하는 역할
        
        ![image.png](image.png)
        
        시그모이드 함수에 통과시켜 어떤 값을 업데이트할지 정함.
        
    
     b. 입력 게이트 : 새로운 정보를 셀 상태에 추가
    
    ![image.png](image%201.png)
    
    시그모이드 함수에 통과시켜 새 정보를 얼마나 받아들일지 결정 
    
     tanh을 이용해 새로 추가될 실제 정보 내용을 만든다.
    
    ![image.png](image%202.png)
    
    새로운 셀 상태를 만든다.
    
    c. 출력 게이트 : 이 셀 상태가 다음 은닉 상태로 출력되도록 함
    
    ![image.png](image%203.png)
    
    $o_t$에서 시그모이드 함수를 통과해 셀 상태를 은닉 상태로 얼마나 출력할지 결정한다.
    
    $h_t$에서 최종 은닉 상태가 만들어진다.
    

### GRU 구조

Gated Recurrent Unit 의 약자.

삭제 게이트와 입력 게이트를 하나의 게이트(z)로 합치고, 셀 상태와 은닉 상태도 합친 구조이다.

![image.png](image%204.png)

[https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)