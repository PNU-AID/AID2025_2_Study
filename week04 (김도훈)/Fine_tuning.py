#pip install scikit-learn으로 패키지 설치 필요
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import numpy as np

# 데이터 불러오기
data = load_wine()
X, y = data.data, data.target

# 데이터 분할
train_input, test_input, train_target, test_target = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 기본 모델 -- 결과 예시[훈련 정확도: 1.000 테스트 정확도: 0.972.] 완벽히 과적합 상태, 모델이 복잡함
rf_default = RandomForestClassifier(random_state=42)
scores = cross_validate(rf_default, train_input, train_target,
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))


#하이퍼파라미터 수정
#1. 트리 깊이 제한 (max_depth) -- 결과예시[훈련 정확도: 1.000 테스트 정확도: 0.972] 약간의 과적합 완화, 일반화 성능 개선.
rf_depth = RandomForestClassifier(max_depth=5, random_state=42, n_jobs=-1)
scores = cross_validate(rf_depth, train_input, train_target,
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))



#2. 트리 개수 조정 (n_estimators) -- 결과예시[훈련 정확도: 0.991 테스트 정확도: 0.969] 트리 수 증가로 안정성이 향상됨.
rf_estimators = RandomForestClassifier(n_estimators=300, max_depth=5,
                                       random_state=42, n_jobs=-1)
scores = cross_validate(rf_estimators, train_input, train_target,
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))



#3. 최소 샘플 분할 수 (min_samples_split) -- 결과예시[훈련 정확도: 0.983 테스트 정확도: 0.974] 과적합 완화 + 테스트 성능 최고점 달성
rf_split = RandomForestClassifier(max_depth=5, n_estimators=300,
                                  min_samples_split=5, random_state=42, n_jobs=-1)
scores = cross_validate(rf_split, train_input, train_target,
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
