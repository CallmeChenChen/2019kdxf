import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA(object):

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def plot_continuous_variable(self, feature, transform_mode=None, fillna_mode=None, label='y', bins=10):
        train = self.train.copy()
        test = self.test.copy()
        if fillna_mode == 'mode':
            mode = train[feature].mode()
            train[feature] = train[feature].fillna(mode)
            test[feature] = test[feature].fillna(mode)
        elif fillna_mode == 'mean':
            avg = train[feature].mean()
            train[feature] = train[feature].fillna(avg)
            test[feature] = test[feature].fillna(avg)
        else:
            train[feature] = train[feature].fillna(-1)
            test[feature] = test[feature].fillna(-1)
        na_value = -1
        if transform_mode == 'log':
            if fillna_mode != 'mode' and fillna_mode != 'mean':
                train1 = train[train[feature] != na_value]
                train2 = train[train[feature] == na_value]
                test1 = test[test[feature] != na_value]
                test2 = test[test[feature] == na_value]
                train1[feature] = np.log(np.abs(train1[feature]) + 1)
                test1[feature] = np.log(np.abs(test1[feature]) + 1)
                train = train1.append(train2)
                test = test1.append(test2)
        elif transform_mode == 'z-score':
            if fillna_mode != 'mode' and fillna_mode != 'mean':
                train1 = train[train[feature] != na_value]
                train2 = train[train[feature] == na_value]
                test1 = test[test[feature] != na_value]
                test2 = test[test[feature] == na_value]
                m = np.mean(train1[feature])
                delta = np.std(train1[feature])
                train1[feature] = train1[feature].apply(lambda x: (x - m) / (delta + 0.0001))
                test1[feature] = test1[feature].apply(lambda x: (x - m) / (delta + 0.0001))
                train = train1.append(train2)
                test = test1.append(test2)
        elif transform_mode == 'min-max':
            if fillna_mode != 'mode' and fillna_mode != 'mean':
                train1 = train[train[feature] != na_value]
                train2 = train[train[feature] == na_value]
                test1 = test[test[feature] != na_value]
                test2 = test[test[feature] == na_value]
                min_value = np.min(train1[feature])
                max_value = np.max(train1[feature])
                train1[feature] = train1[feature].apply(lambda x: (x - min_value) / (max_value - min_value))
                test1[feature] = test1[feature].apply(lambda x: (x - min_value) / (max_value - min_value))
                train = train1.append(train2)
                test = test1.append(test2)
        elif transform_mode == None:
            pass
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.distplot(train.loc[train[label] == 0, feature], label='0', bins=bins)
        sns.distplot(train.loc[train[label] == 1, feature], label='1', bins=bins)
        plt.legend()
        plt.subplot(1, 2, 2)
        sns.distplot(train[feature], label='train')
        sns.distplot(test[feature], label='test')
        plt.legend()

        print("Train and Test describe info:")
        df1 = train[feature].describe().rename('train')
        df2 = test[feature].describe().rename('test')
        result = pd.concat([df1, df2], axis=1)
        print(result)

    def desc_discrete_variables(self, features=None):
        '''字段名称 训练集唯一数 测试集唯一数 相同取值数 overlap训练集长度  overlap测试集长度
            overlap训练集比率 overlap测试集比率
            overlap训练集（取值个数）比率 overlap测试集（取值个数）比率
        '''
        train = self.train.copy()
        test = self.test.copy()

        if features is None:
            features = train.select_dtype(include=['object', 'category']).columns.tolist()
        result = []
        for f in features:
            feat_train_unique = train[f].unique()
            feat_test_unique = test[f].unique()
            # 唯一数
            train_nuniques = len(feat_train_unique)
            test_nuniques = len(feat_test_unique)
            # 相同唯一数
            common_uniques = list(set(feat_train_unique).intersection(feat_test_unique))
            common_nunique = len(common_uniques)
            # overlap 长度
            overlap_train_length = sum(train[f].isin(common_uniques))
            overlap_test_length = sum(test[f].isin(common_uniques))
            # overlap 比率
            overlap_train_ratio = overlap_train_length / len(train)
            overlap_test_ratio = overlap_test_length / len(test)
            # overlap 取值个数 比率
            overlap_uniques_train_ratio = common_nunique / train_nuniques
            overlap_uniques_test_ratio = common_nunique / test_nuniques

            feat_desc = (f, train_nuniques, test_nuniques, common_nunique, overlap_train_length,
                         overlap_test_length, overlap_train_ratio, overlap_test_ratio,
                         overlap_uniques_train_ratio, overlap_uniques_test_ratio)
            result.append(feat_desc)
        cols = ['字段名', '训练集唯一数', '测试集唯一数', '相同取值数', 'overlap训练集长度', 'overlap测试集长度',
                'overlap训练集比率', 'overlap测试集比率', 'overlap训练集(取值个数)比率', 'overlap测试集(取值个数)比率']
        return pd.DataFrame(data=result, columns=cols)

    def desc_single_discrete(self, feature=None, min_count=30, label='label', na_value=-1):
        train = self.train.copy()
        test = self.test.copy()
        cat_nums = train[feature].nunique()
        data = pd.concat([train, test], ignore_index=True)
        data[feature].fillna(na_value, inplace=True)
        print(feature, 'train unique nums', cat_nums, '\n')
        if cat_nums < 50:
            tmp = data.groupby([feature, label])[feature].count().unstack()
            tmp.fillna(0, inplace=True)
            tmp.reset_index(inplace=True)
            tmp['sum'] = tmp[0] + tmp[1]
            target_col = feature + '_' + 'ratio'
            tmp[target_col] = round(tmp[1] / tmp['sum'], 3)
            return tmp[[feature, 'sum', target_col]].sort_values(by='sum', ascending=False)
        else:
            value_counts = train[feature].value_counts()
            filter_value = list(value_counts[value_counts < min_count].index)

            data[feature] = data[feature].replace(list(filter_value), 'UNK')
            tmp = data.groupby([feature, label])[feature].count().unstack()
            tmp.fillna(0, inplace=True)
            tmp.reset_index(inplace=True)
            tmp['sum'] = tmp[0] + tmp[1]
            target_col = feature + '_' + 'ratio'
            tmp[target_col] = round(tmp[1] / tmp['sum'], 3)
            return tmp[[feature, 'sum', target_col]].sort_values(by='sum', ascending=False)

    def plot_dist_custom(self, feature, bins=10, label='label'):
        train = self.train.copy()
        plt.figure(figsize=(8, 6))
        sns.distplot(train.loc[train[label] == 0, feature].dropna(), label='0', bins=bins)
        sns.distplot(train.loc[train[label] == 1, feature].dropna(), label='1', bins=bins)
        plt.legend()

    def calc_feature_psi(self, features=[], is_cate=False, n=10):
        """

        :param features: list[features]
        :param is_cate: True/False
        :param n:
        :return:
        """
        train = self.train.copy().fillna(-999)
        test = self.test.copy().fillna(-999)
        train['group'] = 'train'
        test['group'] = 'test'
        data = pd.concat([train, test], axis=0)
        psi = {}

        if not is_cate:
            for x in features:
                cut_off_p = np.linspace(0, 1, n + 1)
                cut_off = sorted(train[x].quantile(cut_off_p).unique())
                cut_off[0] = -np.inf
                cut_off[-1] = np.inf
                data[x] = pd.cut(data[x], cut_off)
                print('=== cut_off of ', x, ' ===')
                print(cut_off)
                res = data.groupby([x, 'group']).size().unstack().fillna(1)
                res = res / np.sum(res, axis=0)
                print('=== describe of data ===')
                print(res, '\n')
                psi_x = np.sum((res['train'] - res['test']) * np.log(res['train'] / res['test']))
                psi[x] = np.round(psi_x, 4)
        else:
            for x in features:
                res = data.groupby([x, 'group']).size().unstack().fillna(1)
                res = res / np.sum(res, axis=0)
                psi_x = np.sum((res['train'] - res['test']) * np.log(res['train'] / res['test']))
                psi[x] = np.round(psi_x, 4)
        psi_all = pd.DataFrame(columns=['name', 'psi'])
        psi_all['name'] = psi.keys()
        psi_all['psi'] = psi.values()
        return psi_all

    
    
def calc_psi(train, test, features=None, cate_cols=None, n=10):
    """

    :param train:
    :param test:
    :param features: 全部需要计算的特征，包括类别型和连续型
    :param cate_cols: 类别型特征
    :param n:
    :return:
    """
    if features is None:
        features = train.columns.tolist()
    train = train.copy().fillna(-999)
    test = test.copy().fillna(-999)
    train['group'] = 'train'
    test['group'] = 'test'
    data = pd.concat([train, test], axis=0)
    psi = {}

    if cate_cols is None:
        cate_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        cate_cols = None if len(cate_cols) == 0 else cate_cols

    if cate_cols is None:
        cut_off_p = np.linspace(0, 1, n + 1)
        for x in features:
            cut_off = sorted(train[x].quantile(cut_off_p).unique())
            cut_off[0] = -np.inf
            cut_off[-1] = np.inf
            print('=== cut_off of ', x, ' ===')
            print(cut_off)
            data[x] = pd.cut(data[x], cut_off)
            res = data.groupby([x, 'group']).size().unstack().fillna(1)
            res = res / np.sum(res, axis=0)
            print('=== describe of data ===')
            print(res)
            psi_x = np.sum((res['train'] - res['test']) * np.log(res['train'] / res['test']))
            print('===psi of ', x, ' :', psi_x)
            psi[x] = np.round(psi_x, 4)
    else:
        part_features = [x for x in features if x not in cate_cols]
        cut_off_p = np.linspace(0, 1, n + 1)
        for x in part_features:
            cut_off = sorted(train[x].quantile(cut_off_p).unique())
            cut_off[0] = -np.inf
            cut_off[-1] = np.inf
            data[x] = pd.cut(data[x], cut_off)
            print('=== cut_off of ', x, ' ===')
            print(cut_off)
            res = data.groupby([x, 'group']).size().unstack().fillna(1)
            res = res / np.sum(res, axis=0)
            print('=== describe of data ===')
            print(res)
            psi_x = np.sum((res['train'] - res['test']) * np.log(res['train'] / res['test']))
            psi[x] = np.round(psi_x, 4)
        for x in cate_cols:
            res = data.groupby([x, 'group']).size().unstack().fillna(1)
            res = res / np.sum(res, axis=0)
            psi_x = np.sum((res['train'] - res['test']) * np.log(res['train'] / res['test']))
            psi[x] = np.round(psi_x, 4)
    psi_all = pd.DataFrame(columns=['name', 'psi'])
    psi_all['name'] = psi.keys()
    psi_all['psi'] = psi.values()
    return psi_all
