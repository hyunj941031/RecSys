# RecSys

본 Repository에는 추천시스템을 구현한 코드가 저장되어있습니다.

# Used Data
-  FashionCampus
    - 온라인 의류판매 사이트의 거래내역 데이터(출처:https://www.kaggle.com/datasets/latifahhukma/fashion-campus)
    - 모델에 input을 위해 sequence size 통일
    - 샘플링을 통해 학습시간 단축
    - 사용자마다 가장 최근 3회 구매이력을 validation data (기존데이터를 이용하여 향후 구매동향을 분석, 구매 특성상 더 나중에 동향이 나타날 수 있으므로 1회가 아닌 3회로 검증)

# Models
#### Bert4Rec
    - Sequential model: 순서에 따른 연관성을 찾는 모델 (Transformer 기반 모델)
    - Training -> masking을 이용하여 이전,이후의 행동에 대한 학습 (+ noise 데이터를 만들어 모델이 오차에 잘 적응할 수 있게 조절)
    
# Evaluation Method
- validation data : 가장 최근 3개의 item (전체 구매 개수가 3회 이하인 user data 삭제)
1. NDCG@k (Normalized Discounted Cumulative Gain)
    - k개를 추천한 결과로 평가
    - 랭킹이 낮은 아이템이 평가지표에 미치는 영향이 줄어들도록 하는 지표
    - IDCG(이상적인 DCG)로 측정된 DCG를 나눈 값 (0~1)
    $$nDCG_p=\frac{DCG_p}{IDCG_p}$$
    $$DCG_p= \sum_{i=1}^p \frac{rel_i}{\log_2 (i+1)}$$
    $$IDCG_p=\sum_{i=1}^p \frac{rel_i^{opt}}{\log_2 (i+1)}$$
2. HIT Rate@k
    - k개를 추천한 결과로 평가
    - 전체 사용자 수 대비 적중한 사용자 수
    $$HIT Rate=\frac{HIT users}{users}$$

# Evaluation
