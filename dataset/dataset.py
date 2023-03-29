from torch.utils.data import Dataset
import random
import numpy as np
import torch

class BERTTrainDataSet(Dataset):
    def __init__(self, user_train, num_users, num_items, sequence_len=200, mask_prob=0.15, random_seed=None):
        self.user_train = user_train
        self.num_users = num_users
        self.num_items = num_items
        self.sequence_len = sequence_len
        self.mask_prob = mask_prob
        self._all_items = set([i for i in range(1, self.num_items + 1)])

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # tokens
        self.mask_token = self.num_items + 1


    def __len__(self):
        # 총 user 수(학습에 사용할 sequence 수수)
        return self.num_users

    def __getitem__(self, user):
        seq = self.user_train[user]
        tokens = []
        labels = []
        for s in seq[-self.sequence_len:]:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                # BERT 학습
                # random 하게 80% 를 mask token 으로 변환
                if prob < 0.8:
                    # masking
                    # mask_index : num_item + 1 , 0 : pad, 1 ~ num_item : item index
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    # item random sampling (noise)
                    tokens.append(np.random.randint(1, self.num_items))
                else:
                    # 나머지 10% 를 original token 으로 사용
                    tokens.append(s)
                labels.append(s) # 학습에 사용
            # training 에 사용하지 않음 
            else:
                tokens.append(s)
                labels.append(0) # 학습에 사용하지 않음, trivial

        # zero padding 
        # tokens 와 labels 가 padding_len 보다 짧으면 zero padding 을 해준다. 
        padding_len = self.sequence_len - len(tokens)
        
        tokens = [0] * padding_len + tokens
        labels = [0] * padding_len + labels
        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.user_train[user]

    def random_neg_sampling(self, sold_items, num_item_sample):
        nge_samples = random.sample(list(self._all_items - set(sold_items)), num_item_sample)
        return nge_samples

class BertEvalDataset(Dataset):
    def __init__(self, u2seq, u2answer, sequence_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.sequence_len = sequence_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.random_neg_sampling(sold_item=user, num_item_sample = 1)

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.sequence_len:]
        padding_len = self.sequence_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
    
    def random_neg_sampling(self, sold_items, num_item_sample : int):
        nge_samples = random.sample(list(self._all_items - set(sold_items)), num_item_sample)
        return nge_samples