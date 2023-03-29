import math
import numpy as np
import torch
from .utils import ndcg_hit

def train(model, criterion, optimizer, data_loader, device):
    model.train()
    loss_val = 0
    for seq, labels in data_loader:
        logits = model(seq)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1).to(device)  # B*T

        optimizer.zero_grad()
        loss = criterion(logits, labels)

        loss_val += loss.item()

        loss.backward()
        optimizer.step()
    
    loss_val /= len(data_loader)

    return loss_val

def evaluate(model, user_train, user_valid, sequence_len, bert4rec_dataset, make_sequence_dataset, config):
    model.eval()

    NDCG = 0.0 # NDCG@30
    HIT = 0.0 # HIT@30

    idcg = 1/math.log2(2) *3

    users = [user for user in range(make_sequence_dataset.num_users)]

    for user in users:
        seq = ([0]*sequence_len + user_train[user] + [make_sequence_dataset.num_items + 1] * config['val_data'])[-sequence_len:]
        
        with torch.no_grad():
            predictions = model(np.array([seq]))

        ndcg, hit = ndcg_hit(predictions, user_valid[user], 10, idcg)
        
        NDCG += ndcg
        HIT += hit

    NDCG /= len(users)
    HIT /= len(users)

    return NDCG, HIT