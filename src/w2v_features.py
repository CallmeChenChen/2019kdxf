# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
import gc
from gensim.models import Word2Vec
import multiprocessing
import datetime

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


warnings.filterwarnings('ignore')

np.random.seed(993)

def w2c_token_feats(train, test=None, threshold_cnt=15, size=16):
    '''train, valid, test raw data token'''
    train['op'] = 'train'
    test['op'] = 'test'
    data = pd.concat([train, test], axis=0)
    del train
    del test
    gc.collect()
    print("clean model make ...")
    data['model'] = data['model'].str.upper().apply(
        lambda x: x.replace('-', '').replace(',', '').replace('_', '').replace(' ', '').replace('+','')
        if not isinstance(x,float) else x)

    data['make'] = data['make'].str.upper().apply(
        lambda x: x.replace('-', '').replace(',', '').replace('_', '').replace(' ', '').replace('+', '').replace('手机','')
        if not isinstance(x, float) else x)

    if not 'date' in data:
        data['nginxtime'] = (pd.to_datetime(data['nginxtime'], unit='ms')+datetime.timedelta(hours=8)).dt.strftime('%Y-%m-%d %H:%S')
        data['hour'] = data['nginxtime'].apply(lambda x: x.split(' ')[1].split(':')[0])
        data = data.drop(columns=['nginxtime'])
    if not 'hw' in data:
        data['hw'] = data['h'].astype(float) * data['w'].astype(float)

    clean_col = []
    for col in data.columns :
        if col not in ['sid', 'label', 'op', 'date']:
            print("token feat :", col)
            # 不同特征的同一值进行区分
            data[col] = data[col].fillna(col + '_NA').astype(str).replace('empty', col + '_empty')
            data[col] = data[col].replace('nan', col + '_NA')

            if col in ['h', 'w', 'hw', 'apptype', 'province', 'dvctype',
                       'ntt', 'carrier', 'osv', 'orientation']:
                data[col] = data[col].astype(str) + str('_' + col)

            if col in ['hour']:
                data[col + '_token'] = data[col]
                data.drop(columns=[col], inplace=True)
                continue
            # 用train的数据统计 以免过拟合
            tmp = data[col].value_counts()
            idx_list = list(tmp[tmp > threshold_cnt].index)
            #
            data[col + '_token'] = data[col].apply(lambda x: x if x in idx_list else col +'_UNK')
            clean_col.append(col+'_token')
            data.drop(columns=[col], inplace=True)

    token_feats = [col for col in data.columns if '_token' in col]
    # sid + token_feats
    token_data = data[['sid'] + token_feats]
    del data
    gc.collect()
    # word2vec
    model = Word2Vec(token_data[token_feats].values.tolist(), size=size, window=5, min_count=1,
                     workers=multiprocessing.cpu_count(), iter=40)

    new_all = pd.DataFrame()

    for idx, f in enumerate(token_feats):
        print('Start gen feat of: ', f)
        tmp = []
        for v in token_data[f].unique():
            tmp_v = [v]
            tmp_v.extend(model[v])
            tmp.append(tmp_v)
        tmp_df = pd.DataFrame(tmp)
        w2c_list = ['w2c_{0}_{1}'.format(f, n) for n in range(size)]

        tmp_df.columns = [f] + w2c_list # 11

        tmp_df = tmp_df.drop_duplicates()

        tmp_df = token_data[['sid', f]].merge(tmp_df, on=f, how='left')
        tmp_df.drop(columns=[f], inplace=True)

        # print(tmp_df)

        if idx == 0:
            new_all = pd.concat([new_all, tmp_df] , axis=1)
        else:
            new_all = pd.merge(new_all, tmp_df, how='left', on='sid')

    return new_all

def get_w2c(train, test=None, threshold_cnt=15, size=16):
    '''train,test raw data token vocabulary vector
    '''
    train['op'] = 'train'
    test['op'] = 'test'
    data = pd.concat([train, test], axis=0)
    del train
    del test
    gc.collect()
    print("clean model make ...")
    data['model'] = data['model'].str.upper().apply(
        lambda x: x.replace('-', '').replace(',', '').replace('_', '').replace(' ', '').replace('+', '')
        if not isinstance(x, float) else x)

    data['make'] = data['make'].str.upper().apply(
        lambda x: x.replace('-', '').replace(',', '').replace('_', '').replace(' ', '').replace('+', '').replace('手机',
                                                                                                                 '')
        if not isinstance(x, float) else x)

    if not 'date' in data:
        data['time'] = (pd.to_datetime(data['nginxtime'], unit='ms') + datetime.timedelta(hours=8))
        # data['date'] = data['time'].dt.date
        data['hour'] = data['time'].dt.hour
        # data = data.drop(columns=['nginxtime'])
    if not 'hw' in data:
        data['hw'] = data['h'].astype(float) * data['w'].astype(float)
        data['h_w'] = data['h'].astype(str) + data['w'].astype(str)

    clean_col = []
    for col in data.columns:
        if col not in ['sid', 'label', 'op', 'date', 'time', 'nginxtime']:
            print("token feat :", col)
            # 不同特征的同一值进行区分
            data[col] = data[col].fillna(col + '_NA').astype(str).replace('empty', col + '_empty')
            data[col] = data[col].replace('nan', col + '_NA')

            if col in ['h', 'w', 'hw', 'apptype', 'province', 'dvctype',
                       'ntt', 'carrier', 'osv', 'orientation']:
                data[col] = data[col].astype(str) + str('_' + col)

            if col in ['hour']:
                data[col + '_token'] = data[col]
                data.drop(columns=[col], inplace=True)
                continue
            # 用train的数据统计 以免过拟合
            tmp = data[col].value_counts()
            idx_list = list(tmp[tmp > threshold_cnt].index)
            #
            data[col + '_token'] = data[col].apply(lambda x: x if x in idx_list else col + '_UNK')
            clean_col.append(col + '_token')
            data.drop(columns=[col], inplace=True)

    # save train test
    data[data['op']=='train'].to_csv('/cos_person/processed/train_token.csv' , index=False)
    data[data['op']=='test'].drop(columns=['label']).to_csv('/cos_person/processed/test_token.csv' , index=False)
    print(data.columns)

    print('train gensim w2v')

    token_feats = [col for col in data.columns if '_token' in col]
    # sid + token_feats
    token_data = data[['sid'] + token_feats]

    del data
    gc.collect()
    # word2vec
    model = Word2Vec(token_data[token_feats].values.tolist(), size=size, window=5, min_count=1,
                     workers=multiprocessing.cpu_count(), iter=60)
    # print(model.wv.vocab)
    # save word vec
    model.wv.save_word2vec_format('/cos_person/processed/word_vec_32d')

def get_vocab_embeddings(filename, size=32):
    vocabulary = {'UNK':0}
    embeddings = [[0] * size]
    with open(filename, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                print(line)
                continue
            line = line.strip().split(' ')
            word = line[0]
            embed = list(map(lambda x: float(x), line[1:]))
            if word not in vocabulary.keys():
                vocabulary[word] = len(vocabulary)
                embeddings.append(embed)
            else:
                print('word {} is already in vocab')
    return vocabulary, np.asarray(embeddings)

def w2c_ip_feats(data, w2c_feats, size=10):
    data = data.copy()
    for feat in w2c_feats:
        data[feat] = feat + data[feat].astype(str)
    print('train w2c.')
    model = Word2Vec(data[w2c_feats].values.tolist(), size=size, window=100, min_count=1,
                     workers=multiprocessing.cpu_count(), iter=10)
    label_ = data[['sid', 'label']].copy()
    new_all = pd.DataFrame()
    for idx, f in enumerate(w2c_feats):
        print('Start gen feat of: ', f)
        tmp = []
        for v in data[f].unique():
            tmp_v = [v]
            tmp_v.extend(model[v])
            tmp.append(tmp_v)
        tmp_df = pd.DataFrame(tmp)
        w2c_list = ['w2c_{0}_{1}'.format(f, n) for n in range(size)]

        tmp_df.columns = [f] + w2c_list  # 11

        tmp_df = tmp_df.drop_duplicates()

        tmp_df = data[['sid', f]].merge(tmp_df, on=f, how='left')
        tmp_df.drop(columns=[f], inplace=True)

        if idx == 0:
            new_all = pd.concat([new_all, tmp_df], axis=1)
        else:
            new_all = pd.merge(new_all, tmp_df, how='left', on='sid')
    new_all = new_all.merge(label_, on='sid', how='left')
    return new_all


def f1_score_metric(preds, train_set):
    preds = np.where(preds >= 0.499, 1, 0)
    labels = train_set.get_label()
    score = f1_score(y_true=labels, y_pred=preds)
    return 'f1-score', score, True


def lr_decay(current_round):
    lr_start = 0.15
    decay = 0.003
    new_lr = (lr_start * 1.0) / (1.0 + decay * current_round)
    return round(new_lr, 4)
# [lr_decay(i) for i in range(0,3000,500)]

def stack_features(train, test, features, seed_seed=2019):

    params = {
        # 'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 48,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 30,
        'nthread': -1,
        'verbose': -1
    }

    skf = StratifiedKFold(n_splits=5, random_state=seed_seed, shuffle=True)
    # 用于保存中间结果
    train_count_rank = pd.Series()
    # test_count_rank = pd.Series(0, index=list(range(test.shape[0])))
    test_count_rank = np.zeros((test.shape[0], skf.get_n_splits()))
    # 用于保存最后输出结果 sid count_rank_feat
    train_result = train[['sid', 'label']]
    test_result = test[['sid']]
    #
    feature_importance = []

    for n, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['label'])):
        X_train, X_valid = train[features].loc[train_idx], train[features].loc[valid_idx]
        y_train, y_valid = train['label'].loc[train_idx], train['label'].loc[valid_idx]
        data_train = lgb.Dataset(X_train, label=y_train)
        data_valid = lgb.Dataset(X_valid, label=y_valid)
        clf = lgb.train(params,
                        data_train,
                        num_boost_round=5000,
                        valid_sets=[data_train, data_valid],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=200,
                        verbose_eval=50,
                        feval=f1_score_metric,
                        learning_rates=lr_decay)

        train_count_rank = train_count_rank.append(pd.Series(clf.predict(X_valid, num_iteration=clf.best_iteration), index=valid_idx))
        # test_count_rank = test_count_rank + pd.Series(clf.predict(test[features]))
        test_count_rank[:, n] = clf.predict(test[features], num_iteration=clf.best_iteration)

        tmp = {'name': features, 'score': clf.feature_importance(importance_type='gain')}
        feature_importance.append(pd.DataFrame(tmp))

    feature_importance = pd.concat(feature_importance)
    feature_importance = feature_importance.groupby(['name'])['score'].agg('mean').sort_values(ascending=False)
    print('#' * 25)
    print('features rank:\n', feature_importance.head(50))
    print('#' * 25)
    feature_importance.to_csv('/cos_person/processed/stack_feats_importance.csv')
    train_result = pd.merge(train_result, pd.DataFrame(train_count_rank, columns=['ip_stack_feat']), left_index=True, right_index=True)
    test_result['ip_stack_feat'] = test_count_rank.mean(axis=1)

    return train_result, test_result

if __name__ == '__main__':

    print('loading files.')
    use_cols = ['sid', 'ip', 'reqrealip', 'label']
    train_round1 = pd.read_csv('/cos_person/raw/train_round1.csv', usecols=use_cols)
    train_round2 = pd.read_csv('/cos_person/raw/train_round2.csv', usecols=use_cols)
    test = pd.read_csv('/cos_person/raw/test.csv', usecols=use_cols)

    data = pd.concat([train_round1, train_round2, test], ignore_index=True)
    del train_round1, train_round2, test
    gc.collect()

    data['ip'] = data['ip'].apply(lambda x: '9.9.9' if len(x)>15 else x)
    print('processing ip reqrealip.')
    tmp = data['ip'].str.split('.', expand=True)
    tmp.columns = ['ip1', 'ip2', 'ip3', 'ip4']
    data = data.merge(tmp, left_index=True, right_index=True, how='left')

    tmp = data['reqrealip'].str.split('.', expand=True)
    tmp.columns = ['req1', 'req2', 'req3', 'req4']
    data = data.merge(tmp, left_index=True, right_index=True, how='left')

    del tmp
    gc.collect()
    w2c_feats = ['ip1', 'ip2', 'ip3', 'ip4'] + ['req1', 'req2', 'req3', 'req4']
    result = w2c_ip_feats(data, w2c_feats=w2c_feats, size=36)

    print('result columns:', [col for col in  result.columns if 'w2c' not in col])
    print('result shape: ', result.shape)
    # result.to_csv('/cos_person/processed/w2v_ip_feats.csv', index=False)
    train = result[result['label'].notnull()]
    test = result[result['label'].isnull()]
    del result
    gc.collect()

    features = [col for col in train.columns if 'w2c' in col]

    train_stack_feats, test_stack_feats = stack_features(train, test, features=features)

    print(train_stack_feats.columns, test_stack_feats.columns)

    train_stack_feats.to_csv('/cos_person/processed/train_w2c_ip.csv', index=False) # 有label
    test_stack_feats.to_csv('/cos_person/processed/test_w2c_ip.csv', index=False)


    # result = w2c_token_feats(train, test, threshold_cnt=10, size=10)
    # get_w2c(train, test, threshold_cnt=10, size=32)
    # print('result shape: ', result.shape)
    # result.to_csv('/cos_person/processed/w2v_feats.csv', index=False)
    # result.to_csv('F:/2019KDXF/data/processed/w2v_feats_test.csv',index=False)

    # vocab, embeddings = get_vocab_embeddings('/cos_person/processed/word_vec_32d', size=32)

    # import pickle
    # print('vocabulary length is {}'.format(len(vocab)))
    # print('embedding shape:{}'.format(embeddings.shape))
    # print('saving vocab and embedding to pkl files.')
    #
    # with open('/cos_person/processed/vocabulary.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)
    # with open('/cos_person/processed/word_vec_32d.pkl', 'wb') as f:
    #     pickle.dump(embeddings, f)


