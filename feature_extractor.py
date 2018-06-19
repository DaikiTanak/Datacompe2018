import pandas as pd
import numpy as np
from collections import Counter
import hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from chainer.datasets import tuple_dataset

from collections import defaultdict




class Feature_Extractor():


    def __init__(self):
        print("start making features!!!")
        fin = open('data_train.csv')
        #特徴名のリスト
        featname = fin.readline().strip().split(',')

        train = pd.read_csv("data_train.csv")
        test = pd.read_csv("data_test.csv")
        train = train.fillna(0)
        test = test.fillna(0)

        #特徴量として使うid
        features = ["advertiser_id", "category_id", "adnw_id", "adspot_id", "user_type_id"]
        id_list = []
        for f in features:
            e = pd.concat([train[f], test[f]], axis=0)
            id_list.append(e)

        click = np.array(train["click"], dtype="int32")


        #IDをキー、0から始まる新しい割り当てIDを値にもつ辞書dic
        def onehot_dic(ID_list):
            c = Counter(ID_list)
            #IDの種類数
            num = len(c)
            dic = {}
            i = 0
            for ID in c.keys():
                dic[ID] = i
                i += 1
            return dic, num

        def to_onehot(ID_list, dic, num):
            li = []
            for ID in ID_list:
                li.append(dic[ID])
            return np.eye(num)[li]


        dic_lis, num_lis = [], []
        for ele in id_list:
            dic, num = onehot_dic(ele)
            dic_lis.append(dic)
            num_lis.append(num)

        train_li, test_li = [], []

        index = 0
        for f in features:
            train_df = pd.DataFrame(to_onehot(train[f], dic_lis[index], num_lis[index]))
            test_df = pd.DataFrame(to_onehot(test[f], dic_lis[index], num_lis[index]))
            train_li.append(train_df)
            test_li.append(test_df)
            index += 1


        #訓練データ行列とテストデータ行列の作成
        tr_df = pd.concat(train_li, axis=1)
        tes_df = pd.concat(test_li, axis=1)


        X = np.matrix(tr_df)
        X_te = np.matrix(tes_df)

        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)


        decomposer = PCA(n_components=200, random_state=0)
        decomposer.fit(X_scaled)
        X_pca = np.matrix(decomposer.transform(X_scaled), dtype="float32")


        X_te_scaled = scaler.transform(X_te)
        X_te_pca = np.matrix(decomposer.transform(X_te_scaled), dtype="float32")
        self.X_test = X_te_pca

        X_train, X_val ,y_train, y_val = train_test_split(X_pca, click, test_size=0.2, random_state=0)

        train = tuple_dataset.TupleDataset(X_train, y_train)
        val = tuple_dataset.TupleDataset(X_val, y_val)

        self.train = train
        self.val = val


        def make_test(self):
            return self.X_test
