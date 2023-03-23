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
    - 학습을 진행함에 따라 loss 감소, ndcg, hit 증가
    - 하지만, 개선 폭이 크지 않음
<img src = "https://user-images.githubusercontent.com/69951894/227232997-8c3369f1-f6d4-4e7d-8b5c-f1063a8dc875.png" width="50%" height="30%">
<img src = "https://user-images.githubusercontent.com/69951894/227228788-8e189eeb-a316-422a-b753-69cc67b7e70b.png" width="50%" height="30%">
<img src = "https://user-images.githubusercontent.com/69951894/227228853-d4665747-f192-440e-9ed6-a2a5c50f6d62.png" width="50%" height="30%">

- 일부 구매자들에 대한 데이터로 예측한 예시
    - user 1
        - train data: 
    ![image](https://user-images.githubusercontent.com/69951894/227270663-fffbf05e-7a6c-4ecc-8d47-afc88d7993d7.png)
    ![image](https://user-images.githubusercontent.com/69951894/227270835-82a370f8-70eb-4266-83e4-82a2dbf4cf0a.png)
    ![image](https://user-images.githubusercontent.com/69951894/227271015-e94613cd-0ad4-405a-8fce-296a8735c832.png)

        - validation data:
    ![image](https://user-images.githubusercontent.com/69951894/227271156-eabf3dc4-7423-4b34-a35b-cd7f34257ed2.png)
    ![image](https://user-images.githubusercontent.com/69951894/227271271-980827b2-c39e-49e4-af87-93013848e264.png)
    ![image](https://user-images.githubusercontent.com/69951894/227271444-9fa3ed54-fa0e-4487-88c7-5cb0c1c8cd26.png)

        - predicted data:
    ![image](https://user-images.githubusercontent.com/69951894/227271684-74727374-1155-4445-9deb-82bcc36c63b8.png)
    ![image](https://user-images.githubusercontent.com/69951894/227271835-4c1b3e85-33ba-4932-b29b-405e7c708380.png)
    ![image](https://user-images.githubusercontent.com/69951894/227271950-8afa1b43-de86-46ae-a793-40e218b414a3.png)
    ![image](https://user-images.githubusercontent.com/69951894/227272166-0f761aa3-425a-4042-a1e9-8b4e552886e6.png)
    ![image](https://user-images.githubusercontent.com/69951894/227272806-c5d747ff-5b9a-42be-b026-d1efcb038cc8.png)
    ![image](https://user-images.githubusercontent.com/69951894/227273054-0d3e9b50-34bb-4e1f-bf7f-ac59b9a2f8e5.png)
    ![image](https://user-images.githubusercontent.com/69951894/227273264-ecf40aa7-3bf0-4219-8cfc-7e5c132ab501.png)
    ![image](https://user-images.githubusercontent.com/69951894/227273371-e96c155f-c942-4ffa-a348-9a8f011af957.png)
    ![image](https://user-images.githubusercontent.com/69951894/227273498-cc91f053-0a5d-45f6-8952-38a2716537ad.png)
    
    -  user 2
        - train data: 
    ![image](https://user-images.githubusercontent.com/69951894/227284031-5a5efa3d-3de4-465b-a483-54729907ab29.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284056-284ecaeb-1a40-4366-8559-54f7d80ec566.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284083-a3623f97-ab81-4dfa-be31-d052f19e24fc.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284121-a59bbc95-2860-4aa7-afaa-aff0b87f2579.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284148-17d3896b-2534-43bf-9fe4-8be94328bad5.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284187-2001a338-0353-443d-a2c8-3afc310c1bbd.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284214-b7c924ac-f0cf-4878-aaea-188d6e0b9fc3.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284249-df801b7f-804b-4a00-ae56-806da5bdd573.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284274-d50d3b63-0604-49af-af45-f0e35852fe41.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284302-49bb1ae4-0132-473e-80ff-7e328c0901ab.png)

        - validation data:
    ![image](https://user-images.githubusercontent.com/69951894/227284762-797f1214-1ca8-4fdc-ae73-c5884b08b7d3.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284788-ed13ce4a-0cfe-4a83-8e2b-80deea3fbb74.png)
    ![image](https://user-images.githubusercontent.com/69951894/227284808-be353066-400d-47ed-b86f-fce7272fd7ed.png)

        - predicted data:
    ![image](https://user-images.githubusercontent.com/69951894/227282175-6377df62-0b65-41de-80c4-84626fc35ec0.png)

    
# Conclusion
    - Item Id를 유사도가 비슷한(의류의 종류, 가격, 색상, 계절옷, 브랜드 등) Item끼리 묶어서 Id를 재배열하면 더 좋은 성능을 발현할 수 있을 것으로 판단된다.
    - 한정적인 컴퓨터 자원으로 인해 모델 구조의 변경이나 파라미터 최적화의 기회가 많지 않았다.
    - validation data를 가장 마지막에 구매한 이력 1회를 고려했던 것 보다 3회를 고려했을 때 성능이 좋게 나왔고 학습이 보다 원활히 이루어졌다.
