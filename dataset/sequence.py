import pandas as pd
import os
from collections import defaultdict

class MakeSequenceDataSet():
    """
    SequenceData 생성
    """
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(os.path.join(self.config.data_path, 'user_item.csv'), index_col=0)
        if config['delete_data']:
            self.df = self.delete_ones()
        if config['sampling']:
            self.df = self.sampling_users()

        self.item_encoder, self.item_decoder = self.generate_encoder_decoder('itemId')
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder('userId')
        self.num_items, self.num_users = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['itemId'].apply(lambda x : self.item_encoder[x] + 1)
        self.df['user_idx'] = self.df['userId'].apply(lambda x : self.user_encoder[x])
        self.df = self.df.sort_values(['user_idx', 'timestamp']) # 시간에 따른 정렬
        self.user_train, self.user_valid = self.generate_sequence_data()


    def delete_ones(self):
        a = self.df.groupby('userId')['itemId'].size()
        for i in a.index:
            if a[i] <= self.config['val_data']:
                del(a[i])
        df_ = self.df.copy()
        df_ = df_[df_['userId'].isin(a.index)]
                
        return df_


    def sampling_users(self):
        a = self.df.groupby('userId')['userId'].mean().sample(frac=self.config['sampling_frac'], random_state=self.config['random_seed'])
        df_ = self.df.copy()
        df_ = df_[df_['userId'].isin(a.index)]
        return df_


    def generate_encoder_decoder(self, col:str) -> dict:
        '''
        encoder, decoder 생성

        Args:
            col (str): 생성할 columns 명
        Return:
            dict: 생성된 user encoder, decoder
        '''
        encoder = {}
        decoder = {}
        ids = self.df[col].unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder


    def generate_sequence_data(self) -> dict:
        '''
        sequence data 생성

        Return:
            dict: train user sequence / valid user sequence
        '''
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        group_df = self.df.groupby('user_idx')
        for user, item in group_df:
            users[user].extend(item['item_idx'].tolist())

        for user in users:
            user_train[user] = users[user][:-self.config['val_data']]
            user_valid[user] = users[user][-self.config['val_data']:] # 마지막 아이템 예측

        return user_train, user_valid


    def get_train_valid_data(self):
        return self.user_train, self.user_valid