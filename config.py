import torch

config = {
    'data_path' : '/content/drive/MyDrive/fashion_campus_dataset',
    
    'sequence_len' : 50,
    'mask_prob' : 0.3, # cloze task
    'random_seed' : 123,

    'num_layers' : 2, # block 수 (encoder layer 수)
    'hidden_units' : 50, # Embedding size
    'num_heads' : 2, # Multi-head layer 수 (병렬처리), hidden_units를 나눴을 때 나누어 떨어지게게
    'dropout' : 0.15, # dropout의 비율

    'epoch' : 5,
    'patience' : 5,
    'batch_size' : 256,
    'lr' : 0.001,

    'num_epochs' : 10,
    'num_workers' : 4,
    'val_data' : 3,
    'delete_data' : True,
    'sampling' : True,
    'sampling_frac' : 0.25,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'