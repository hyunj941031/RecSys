import math

def ndcg_hit(predict, valid, k, idcg, config):
    predict = predict.cpu()
    valid = valid
    temp = (-predict[0][-config["val_data"]:]).argsort(dim=1)[:, :k].numpy()

    dcg = 0
    hit = 0
    for t in range(config['val_data']):
        for i, r in enumerate(temp[t]):
            if r in valid:
                dcg += 1 / math.log2(i + 2)
                hit = 1

    ndcg = dcg / idcg
    return ndcg, hit