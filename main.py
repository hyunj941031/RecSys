import dataset
from torch.utils.data import DataLoader
from models.bert_model import BERT4Rec
from trainers import train, evaluate
from config import config, device

import torch
import torch.nn as nn
from tqdm import tqdm
from box import Box

import warnings

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=True)

config = Box(config)

def run():
    make_sequence_dataset = dataset.MakeSequenceDataSet(config=config)
    user_train, user_valid = make_sequence_dataset.get_train_valid_data()
    
    bert4rec_dataset = dataset.BERTTrainDataSet(
            user_train = user_train, 
            sequence_len = config.sequence_len, 
            num_users = make_sequence_dataset.num_users, 
            num_items = make_sequence_dataset.num_items,
            mask_prob = config.mask_prob,
            )

    data_loader = DataLoader(
            bert4rec_dataset, 
            batch_size = config.batch_size, 
            shuffle = True, 
            pin_memory = True,
            num_workers = config.num_workers,
            )
    
    model = BERT4Rec(
            num_users = make_sequence_dataset.num_users, 
            num_items = make_sequence_dataset.num_items, 
            hidden_units = config.hidden_units, 
            num_heads = config.num_heads, 
            num_layers = config.num_layers, 
            sequence_len = config.sequence_len, 
            dropout = config.dropout, 
            device = device,
            ).to(device)
        
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    loss_list = []
    ndcg_list = []
    hit_list = []
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        tbar = tqdm(data_loader)
        for _ in tbar:
            train_loss = train(
                model = model, 
                criterion = criterion, 
                optimizer = optimizer, 
                data_loader = data_loader,
                device
                )
            
            ndcg, hit = evaluate(
                model = model, 
                user_train = user_train, 
                user_valid = user_valid, 
                sequence_len = config.sequence_len,
                bert4rec_dataset = bert4rec_dataset, 
                make_sequence_dataset = make_sequence_dataset,
                config
                )

            loss_list.append(train_loss)
            ndcg_list.append(ndcg)
            hit_list.append(hit)

            tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@30: {ndcg:.5f}| HIT@30: {hit:.5f}')
            
if __name__ == '__main__':
    run()