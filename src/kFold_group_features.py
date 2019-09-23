from sklearn.model_selection import StratifiedKFold
import category_encoders as ce

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



def kFold_encoder_features(train, test, features, seed_seed=2019):
    print("k-fold ce encoder ...")
    train = train.copy()
    test = test.copy()
    kfold = StratifiedKFold(n_splits=5, random_state=seed_seed, shuffle=True)
    encoder = ce.CatBoostEncoder(cols=features)
    # encoder = ce.WOEEncoder(cols=[feat])
    # ce 默认nan用label.mean填充
    train[features] = train[features].astype(str)
    test[features] = test[features].astype(str)
    # test
    encoder.fit(train, train['label'])
    test = encoder.transform(test)
    test[features] = test[features].astype(np.float32)
    # train
    feat_encoder = train.copy()

    for n_fold, (train_idx, valid_idx) in enumerate(kfold.split(train, train['label'])):
        print('processing fold: ', n_fold+1)
        encoder.fit(train.loc[train_idx], train.loc[train_idx, 'label'])

        v_df = encoder.transform(train.loc[valid_idx]) #df

        feat_encoder.loc[valid_idx, features] = v_df[features].values

    train[features] = feat_encoder[features].astype(np.float32)
    print('ce encoder done.')

    return pd.concat([train, test], ignore_index=True)


def kFold_encoder_features2(train, test, features, seed_seed=2019):
    print("k-fold ce encoder ...")
    kfold = StratifiedKFold(n_splits=5, random_state=seed_seed, shuffle=True)
    for feat in tqdm(features):
        encoder = ce.CatBoostEncoder(cols=[feat])
        # encoder = ce.WOEEncoder(cols=[feat])
        train[feat] = train[feat].astype(str)
        test[feat] = test[feat].astype(str)
        # test
        encoder.fit(train[feat], train['label'])
        test[feat] = encoder.transform(test[feat]).values.reshape(-1)
        # train
        feat_encoder = train[[feat]].copy()
        for n_fold, (train_idx, valid_idx) in enumerate(kfold.split(train, train['label'])):
            encoder.fit(train.loc[train_idx, feat], train.loc[train_idx, 'label'])
            v = encoder.transform(train.loc[valid_idx, feat])
            feat_encoder.loc[valid_idx, feat] = v.values.reshape(-1)
        train[feat] = feat_encoder[feat].astype(np.float32)
    print('ce encoder done.')
    return train, test
