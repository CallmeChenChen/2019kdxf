from sklearn.model_selection import StratifiedKFold

def kFold_group_feats(train, test, group_feats=None, seed_seed=2019):
    '''
    折内统计
    :param train: 线下训练集或者全量训练集
    :param test: 线下验证集或者线上测试集
    :param seed_seed:
    :return:
    '''
    skf = StratifiedKFold(n_splits=5, random_state=seed_seed)
    train['fold'] = None
    for n_fold, (train_index, valid_index) in enumerate(skf.split(train, train)):
        train.loc[valid_index, 'fold'] = n_fold

    for feat1 in group_feats:
        for feat2 in ['ratio','h','w']:
            new_col = feat1 + '_gp_' + feat2 + '_kfold_mean'
            # test
            test[new_col] = test.groupby(feat1)[feat2].mean()
            # train k fold
            for n_fold, (train_index, valid_index) in enumerate(skf.split(train, train)):
                tmp_train = train.loc[train_index]
                fold_result = tmp_train.groupby(feat1)[feat2].mean()
                tmp = train.loc[train['fold']==n_fold, [feat1]]
                train.loc[train['fold']==n_fold, new_col] = tmp[feat1].map(fold_result)
                # fillna
                media = tmp_train[feat2].media()
                train.loc[train['fold']==n_fold, new_col] = train.loc[train['fold']==n_fold, new_col].fillna(media)
    train.drop(columns=['fold'], inplace=True)
    return train, test
