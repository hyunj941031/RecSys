# RecSys

본 Repository에는 추천시스템을 구현한 코드가 저장되어있습니다.

# Used Data
- FashionCampus

# Models
- Bert4Rec

# Evaluation
1. NDCG@k (Normalized Discounted Cumulative Gain)
    - k개를 추천한 결과로 평가
    - 랭킹이 낮은 아이템이 평가지표에 미치는 영향이 줄어들도록 하는 지표
    - IDCG(이상적인 DCG)로 측정된 DCG를 나눈 값 (0~1)
    - $$nDCG_p=\frac{DCG_p}{IDCG_p}$$
    - $$DCG_p= \sum_{i=1}^p \frac{rel_i}{\log_2 (i+1)}$$
    - $$IDCG_p=\sum_{i=1}^p \frac{rel_i^{opt}}{\log_2 (i+1)}$$
