import xgboost as xgb
import pandas as pd
import random
import math
from collections import defaultdict
from dataclasses import dataclass
import ahocorasick
from statsmodels.stats.proportion import proportion_confint
from sklearn.tree import DecisionTreeClassifier

@dataclass
class FilteredGroup:
    key: str
    label: int
    size: int
    group: list


class CoreTabXGB:
    def __init__(self, trees_number=30, sample_percent=0.03, examples_to_keep=10_000,  threshold=1,
                 params={'objective': 'binary:logistic'},
                 index_name='index', n_jobs=24):
        self.trees_number = trees_number
        self.sample_percent = sample_percent
        self.params = params
        self.params.update({'n_jobs': n_jobs})
        self.index_name = index_name
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.examples_to_keep = examples_to_keep
        self.threshold = threshold

        self.model = None
        self.A = None
        self.X_leaves = None
        self.groups = None
        self.target_col = 'target_col'

    def get_dmatrix(self, X, y=None):
        if self.params.get('enable_categorical', False):
            if y is not None:
                dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True)
            else:
                dmatrix = xgb.DMatrix(X, enable_categorical=True)
        else:
            if y is not None:
                dmatrix = xgb.DMatrix(X, label=y)
            else:
                dmatrix = xgb.DMatrix(X)
        return dmatrix

    def filter_groups(self, i):
        new_groups = defaultdict(list)

        if self.groups is None:
            groups_first_leaf = self.X_leaves.reset_index().loc[:, ['index', 'leaf_0', self.target_col]].groupby(
                'leaf_0')
            new_groups = {str(leaf): list(zip(list(group['index'].values), list(group[self.target_col].values))) for
                          leaf, group in groups_first_leaf}
        else:
            index_to_leaf = self.X_leaves[f'leaf_{i}'].to_dict()
            for key, group in self.groups.items():
                if len(group) <= self.threshold:
                    continue
                for item in group:
                    new_groups[f'{key}_{index_to_leaf[item[0]]}'].append(item)
        indexes_to_drop = []
        groups_to_drop = []

        for key, group in new_groups.items():
            labels_sum = sum(item[1] for item in group)
            group_length = len(group)
            if group_length <= self.threshold:
                continue
            elif labels_sum == group_length or labels_sum == 0:
                self.hom_groups_candidates.append(FilteredGroup(key=key,
                                                                label=group[0][1],
                                                                size=group_length,
                                                                group=group))
                indexes_to_drop = indexes_to_drop + [item[0] for item in group]
                groups_to_drop.append(key)

        for key in groups_to_drop:
            new_groups.pop(key)

        self.groups = new_groups
        self.X_leaves = self.X_leaves[~self.X_leaves.index.isin(indexes_to_drop)]

        return indexes_to_drop

    def choose_groups(self, y_train):
        sorted_candidates = sorted(self.hom_groups_candidates, key=lambda g: g.size, reverse=True)

        label_counter = defaultdict(int)
        label_amount = dict(y_train.value_counts())
        indexes_to_filter = []
        for group in sorted_candidates:
            label_counter[group.label] += group.size
            if label_amount[group.label] - label_counter[group.label] < self.examples_to_keep:
                continue
            else:
                self.hom_groups[group.key] = group.label
                candidate_indexes = [item[0] for item in group.group]
                new_indexes_to_filter = random.sample([item[0] for item in group.group],
                                                      k=math.floor((1 - self.sample_percent) * len(candidate_indexes)))
                indexes_to_filter = indexes_to_filter + new_indexes_to_filter

        return indexes_to_filter

    def get_guarantees(self, X_test, y_test, confidence=0.8):

        pred_leaves = self.model.predict(self.get_dmatrix(X_test), pred_leaf=True)
        groups_mistakes_dict = {key: 0 for key in self.hom_groups.keys()}
        groups_candidates_dict = {g.key: g for g in self.hom_groups_candidates}
        groups_dict = {key: groups_candidates_dict[key] for key in self.hom_groups.keys()}
        for pred_lst, label in zip(pred_leaves, y_test):
            joined = '_'.join([str(s) for s in pred_lst])
            key = self._is_in_homogeneous_group(joined)
            if key is not None and label != groups_dict[key].label:
                groups_mistakes_dict[key] += 1

        groups_lst = [(groups_dict[key], groups_mistakes_dict[key]) for key in groups_dict.keys()]
        groups_0 = [(g[1], g[0].size, g[0].label) for g in groups_lst if g[0].label == 0]
        groups_1 = [(g[1], g[0].size, g[0].label) for g in groups_lst if g[0].label == 1]
        train_size_1 = self.original_sizes[1]
        train_size_0 = self.original_sizes[0]
        if groups_0:
            mistakes_0_df = pd.DataFrame(data=groups_0,
                                         columns=['group_mistakes', 'group_size', 'label'])
            mistakes_0_df['total_mistakes'] = mistakes_0_df['group_mistakes'].cumsum()
            mistakes_0_df['total_size'] = mistakes_0_df['group_size'].cumsum()
            mistakes_0_df['percent_remained'] = (train_size_0 - mistakes_0_df['total_size']) / train_size_0
            mistakes_0_df['delta_recall'] = mistakes_0_df['total_mistakes'].apply(
                lambda x: proportion_confint(count=x, nobs=len(y_test[y_test == 1]),
                                             alpha=1-confidence, method='wilson')[1])
        else:
            mistakes_0_df = None

        if groups_1:
            mistakes_1_df = pd.DataFrame(data=groups_1,
                                         columns=['group_mistakes', 'group_size', 'label'])
            mistakes_1_df['total_mistakes'] = mistakes_1_df['group_mistakes'].cumsum()
            mistakes_1_df['total_size'] = mistakes_1_df['group_size'].cumsum()
            mistakes_1_df['percent_remained'] = (train_size_1 - mistakes_1_df['total_size']) / train_size_1
            pn_ratio = train_size_1 / train_size_0
            mistakes_1_df['delta_precision'] = mistakes_1_df['total_mistakes'].apply(
                lambda x: 1 / (1 + pn_ratio / proportion_confint(count=x, nobs=len(y_test[y_test == 0]),
                                                    alpha=1-confidence, method='wilson'))[1])
        else:
            mistakes_1_df = None

        return mistakes_0_df, mistakes_1_df

    def _is_in_homogeneous_group(self, joined):
        occurrences = [i for i in range(len(joined)) if joined.startswith('_', i)]
        hom_comb = [joined[:i] for i in occurrences if self.A.exists(joined[:i])]
        if hom_comb:
            return hom_comb[0]
        return None

    def _predict_by_leafs(self, joined, pred_value):
        occurrences = [i for i in range(len(joined)) if joined.startswith('_', i)]
        target_values = [self.hom_groups[joined[:i]] for i in occurrences if self.A.exists(joined[:i])]
        if target_values:
            return target_values[0]
        else:
            return round(pred_value)

    def filter_df_and_dmatrix(self, dmatrix: xgb.DMatrix, data: pd.DataFrame, indexes, index_name='index'):
        new_data = data.reset_index()
        filtered_data = new_data[new_data[index_name].isin(indexes)]
        slice_indexes = list(filtered_data.index)
        filtered_dmatrix = dmatrix.slice(slice_indexes)
        return filtered_data.set_index(index_name), filtered_dmatrix

    def create_coreset(self, X_train: pd.DataFrame, y_train):

        self.X_leaves = pd.DataFrame(y_train.values, columns=[self.target_col], index=y_train.index)
        self.hom_groups = dict()
        self.groups = None

        iter_dtrain = self.get_dmatrix(X_train, y_train)
        self.model = xgb.train(self.params, num_boost_round=self.trees_number, dtrain=iter_dtrain)
        pred_leaves = self.model[: self.trees_number].predict(iter_dtrain, pred_leaf=True)

        self.X_leaves = self.X_leaves.assign(**{f'leaf_{i}': pred_leaves[:, i] for i in range(self.trees_number)})
        for i in range(self.trees_number):
            self.filter_groups(i)

        indexes_to_filter = self.choose_groups(y_train)
        indexes_to_keep = sorted(set(X_train.index).difference(set(indexes_to_filter)))
        X_train_filtered, iter_dtrain = self.filter_df_and_dmatrix(iter_dtrain, X_train, indexes_to_keep)
        y_train_filtered = y_train[X_train_filtered.index]

        self.A = ahocorasick.Automaton()
        for key, target in self.hom_groups.items():
            self.A.add_word(key, (target, key))

        return X_train_filtered, y_train_filtered


class CoreTabDT:

    def __init__(self, sample_percent=0.03, examples_to_keep=1000, params=None, index_name='index'):

        self.sample_percent = sample_percent
        self.params = params
        self.index_name = index_name
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.examples_to_keep = examples_to_keep
        self.results = []

        self.model = None
        self.X_leaves = None
        self.groups = None
        self.target_col = 'target_col'

    def choose_groups(self, y_train):
        sorted_candidates = sorted(self.hom_groups_candidates, key=lambda g: g.size, reverse=True)

        label_counter = defaultdict(int)
        label_amount = dict(y_train.value_counts())
        indexes_to_filter = []

        for group in sorted_candidates:
            label_counter[group.label] += group.size
            if label_amount[group.label] - label_counter[group.label] < self.examples_to_keep:
                continue
            else:
                self.hom_groups[group.key] = group.label
                candidate_indexes = [item for item in group.group]
                new_indexes_to_filter = random.sample([item for item in group.group],
                                                      k=math.floor((1 - self.sample_percent) * len(candidate_indexes)))
                indexes_to_filter = indexes_to_filter + new_indexes_to_filter

        return indexes_to_filter

    def get_guarantees(self, X_test, y_test, confidence=0.8):
        pred_leaves = self.model.apply(X_test)
        groups_mistakes_dict = {key: 0 for key in self.hom_groups.keys()}
        groups_candidates_dict = {g.key: g for g in self.hom_groups_candidates}
        groups_dict = {key: groups_candidates_dict[key] for key in self.hom_groups.keys()}
        for pred_leaf, label in zip(pred_leaves, y_test):
            key = self._is_in_homogeneous_group(pred_leaf)
            if key is not None and label != groups_dict[key].label:
                groups_mistakes_dict[key] += 1

        groups_lst = [(groups_dict[key], groups_mistakes_dict[key]) for key in groups_dict.keys()]
        groups_0 = [(g[1], g[0].size, g[0].label) for g in groups_lst if g[0].label == 0]
        groups_1 = [(g[1], g[0].size, g[0].label) for g in groups_lst if g[0].label == 1]
        train_size_1 = self.original_sizes[1]
        train_size_0 = self.original_sizes[0]
        if groups_0:
            mistakes_0_df = pd.DataFrame(data=groups_0,
                                         columns=['group_mistakes', 'group_size', 'label'])
            mistakes_0_df['total_mistakes'] = mistakes_0_df['group_mistakes'].cumsum()
            mistakes_0_df['total_size'] = mistakes_0_df['group_size'].cumsum()
            mistakes_0_df['percent_remained'] = (train_size_0 - mistakes_0_df['total_size']) / train_size_0
            mistakes_0_df['delta_recall'] = mistakes_0_df['total_mistakes'].apply(
                lambda x: proportion_confint(count=x, nobs=len(y_test[y_test == 1]),
                                             alpha=1 - confidence, method='wilson')[1])
        else:
            mistakes_0_df = None

        if groups_1:
            mistakes_1_df = pd.DataFrame(data=groups_1,
                                         columns=['group_mistakes', 'group_size', 'label'])
            mistakes_1_df['total_mistakes'] = mistakes_1_df['group_mistakes'].cumsum()
            mistakes_1_df['total_size'] = mistakes_1_df['group_size'].cumsum()
            mistakes_1_df['percent_remained'] = (train_size_1 - mistakes_1_df['total_size']) / train_size_1
            pn_ratio = train_size_1 / train_size_0
            mistakes_1_df['delta_precision'] = mistakes_1_df['total_mistakes'].apply(
                lambda x: 1 / (1 + pn_ratio / proportion_confint(count=x, nobs=len(y_test[y_test == 0]),
                                                                 alpha=1 - confidence, method='wilson'))[1])
        else:
            mistakes_1_df = None

        return mistakes_0_df, mistakes_1_df

    def _is_in_homogeneous_group(self, leaf_id):
        target_value = self.hom_groups.get(leaf_id, None)
        if target_value is None:
            return None
        return leaf_id

    def _predict_by_leafs(self, leaf_id, pred_value):
        target_value = self.hom_groups.get(leaf_id, None)
        if target_value is None:
            return round(pred_value)
        else:
            return target_value

    def te_predict(self, X_test, prediction_model):
        pred_leaves = self.model.apply(X_test)
        pred_leaves_df = pd.DataFrame(pred_leaves, index=X_test.index, columns=['leaf_id'])

        if prediction_model is not None:
            preds = prediction_model.predict(X_test)
        else:
            raise
        pred_leaves_df.loc[:, 'pred_value'] = preds

        predictions = pred_leaves_df.apply(lambda row: self._predict_by_leafs(row['leaf_id'], row['pred_value']),
                                           axis=1)
        return predictions

    def create_coreset(self, X_train: pd.DataFrame, y_train):

        self.X_leaves = pd.DataFrame(y_train.values, columns=[self.target_col], index=y_train.index)
        self.hom_groups = dict()
        self.groups = None

        if not self.params:
            self.model = DecisionTreeClassifier()
        else:
            self.model = DecisionTreeClassifier(**self.params)

        self.model.fit(X_train, y_train)
        pred_leaves = self.model.apply(X_train)
        self.X_leaves.loc[:, 'leaf_id'] = pred_leaves

        groups = self.X_leaves.groupby(['leaf_id']).agg(['sum', 'count'])[self.target_col].sort_values('count',
                                                                                                       ascending=False).reset_index()
        groups.loc[:, 'target_col'] = groups.apply(
            lambda row: int(row['sum'] / row['count']) if row['sum'] == 0 or row['sum'] == row['count'] else None, axis=1)
        groups.dropna(inplace=True)
        self.hom_groups_candidates = [FilteredGroup(key=row['leaf_id'], label=row['target_col'], size=row['count'],
                                                    group=self.X_leaves[
                                                        self.X_leaves['leaf_id'] == row['leaf_id']].index.tolist()) for
                                      index, row in groups.iterrows()]
        indexes_to_filter = self.choose_groups(dataset)
        indexes_to_keep = sorted(set(X_train.index).difference(set(indexes_to_filter)))
        X_train_filtered = X_train.loc[indexes_to_keep, :]
        y_train_filtered = y_train[X_train_filtered.index]

        return X_train_filtered, y_train_filtered

