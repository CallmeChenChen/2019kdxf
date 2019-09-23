# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import datetime
import re
import gc

np.random.seed(993)


def f1_score_metric(preds, train_set):
    preds = np.where(preds > 0.5, 1, 0)
    labels = train_set.get_label()
    score = f1_score(y_true=labels, y_pred=preds)
    return 'f1-score', score, True


def stack_features(train, test, features, seed_seed=2019):
    ''' 用于拟合 count rank'''
    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 32,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
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
                        verbose_eval=20,
                        feval=f1_score_metric,
                        categorical_feature=None)

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
    train_result = pd.merge(train_result, pd.DataFrame(train_count_rank, columns=['count_rank_feat']), left_index=True, right_index=True)
    test_result['count_rank_feat'] = test_count_rank.mean(axis=1)
    return train_result, test_result
