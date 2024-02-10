import time
import xgboost as xgb
import pandas as pd
import random
import math
from collections import defaultdict
from dataclasses import dataclass
import ahocorasick
from statsmodels.stats.proportion import proportion_confint
from sklearn.tree import DecisionTreeClassifier

from src.dataset import Dataset
from .filtering_base import FilteringExperiment, FilteringResults


class NoneFilter(FilteringExperiment):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model
        self.results = []

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        self.model.fit(dataset.X_train, dataset.y_train)
        run_time = time.time() - sx
        score = dataset.metric(dataset.y_test, self.model.predict(dataset.X_test))
        results = FilteringResults(score=score,
                                   run_time=run_time,
                                   filtered_score=score,
                                   new_size_percent=1)
        return results

class RandomSampleFilter(FilteringExperiment):
    def __init__(self, name, p, model):
        super().__init__(name)
        self.p = p
        self.model=model

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        dataset.X_train[dataset.target_col] = dataset.y_train
        X_real_train = dataset.X_train.groupby(dataset.target_col).sample(frac=self.p)
        y_real_train = dataset.y_train[X_real_train.index]

        dataset.X_train.drop(columns=dataset.target_col, inplace=True)
        X_real_train.drop(columns=dataset.target_col, inplace=True)

        self.model.fit(X_real_train, y_real_train)
        run_time = time.time() - s
        score = dataset.metric(dataset.y_test, self.model.predict(dataset.X_test))
        results = FilteringResults(score=score,
                                   run_time=run_time,
                                   filtered_score=score,
                                   new_size_percent=len(X_real_train) / len(dataset.X_train))
        return results


class XgboostSubsampleFilter(FilteringExperiment):

    def __init__(self, p):
        self.p = p

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        model = xgb.XGBClassifier(subsample=self.p, random_state=random.randint(1, 30))
        X_real_train = dataset.X_train
        y_real_train = dataset.y_train[X_real_train.index]
        model.fit(X_real_train, y_real_train)

        run_time = time.time() - s
        score = dataset.metric(dataset.y_test, self.model.predict(dataset.X_test))
        results = FilteringResults(score=score,
                                   run_time=run_time,
                                   filtered_score=score,
                                   new_size_percent=len(X_real_train) / len(dataset.X_train))
        return results




@dataclass
class FilteredGroup:
    key: str
    label: int
    size: int
    group: list


class FilterEachIterXgboostPathSampleFinal(FilteringExperiment):

    def __init__(self, name,
                 trees_number=100, sample_percent=0, examples_to_keep=1000, prediction_model=None, trees_to_stop=None,
                 params={'objective': 'binary:logistic'},
                 index_name='index', threshold=40, n_jobs=24):
        super().__init__(name)
        self.model = None
        self.prediction_model = prediction_model if prediction_model is not None else None
        self.trees_number = trees_number
        self.sample_percent = sample_percent
        self.params = params
        self.params.update({'n_jobs': n_jobs})
        self.trees_to_stop = trees_to_stop
        self.index_name = index_name
        self.X_leaves = None
        self.groups = None
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.examples_to_keep = examples_to_keep
        self.threshold = threshold
        self.results = []
        self.A = None
        self.X_train_filtered = None
        self.y_train_filtered = None
        self.original_sizes = None

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

    def filter_groups(self, dataset: Dataset, i):
        new_groups = defaultdict(list)

        if self.groups is None:
            groups_first_leaf = self.X_leaves.reset_index().loc[:, ['index', 'leaf_0', dataset.target_col]].groupby(
                'leaf_0')
            new_groups = {str(leaf): list(zip(list(group['index'].values), list(group[dataset.target_col].values))) for
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

    def choose_groups(self, dataset: Dataset):
        sorted_candidates = sorted(self.hom_groups_candidates, key=lambda g: g.size, reverse=True)

        label_counter = defaultdict(int)
        label_amount = dict(dataset.y_train.value_counts())
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
        # pred_leaves = self.model.predict(xgb.DMatrix(data=(X_test)), pred_leaf=True)
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

    def get_predictions_df(self, X_test):
        pred_leaves = self.model.predict(xgb.DMatrix(data=(X_test)), pred_leaf=True)
        pred_leaves_df = pd.DataFrame(pred_leaves, index=X_test.index,
                                      columns=[f'leaf_{i}' for i in range(len(self.model.get_dump()))])

        if self.prediction_model is None:
            preds = self.model.predict(xgb.DMatrix(data=X_test))
        else:
            preds = self.prediction_model.predict(X_test)
        pred_leaves_df.loc[:, 'pred_value'] = preds

        pred_leaves_df.loc[:, 'joined'] = pred_leaves_df.apply(
            lambda row: '_'.join([str(row[c]) for c in pred_leaves_df.columns if c.startswith('leaf')]), axis=1)
        pred_leaves_df.loc[:, 'pred_by_groups'] = pred_leaves_df.apply(lambda row: self._is_in_homogeneous_group(row['joined']),
                                                                       axis=1)
        pred_leaves_df.loc[:, 'final_prediction'] = pred_leaves_df.apply(
            lambda row: self._predict_by_leafs(row['joined'], row['pred_value']), axis=1)
        return pred_leaves_df

    def filter_df_and_dmatrix(self, dmatrix: xgb.DMatrix, data: pd.DataFrame, indexes, index_name='index'):
        new_data = data.reset_index()
        filtered_data = new_data[new_data[index_name].isin(indexes)]
        slice_indexes = list(filtered_data.index)
        filtered_dmatrix = dmatrix.slice(slice_indexes)
        return filtered_data.set_index(index_name), filtered_dmatrix

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        X_real_train = dataset.X_train.sort_index()
        y_real_train = dataset.y_train.sort_index()
        self.original_sizes = dict(y_real_train.value_counts())
        self.X_leaves = pd.DataFrame(y_real_train)
        self.hom_groups = dict()
        self.groups = None

        iter_dtrain = self.get_dmatrix(X_real_train, y_real_train)
        self.model = xgb.train(self.params, num_boost_round=self.trees_to_stop, dtrain=iter_dtrain)
        pred_leaves = self.model[: self.trees_to_stop].predict(iter_dtrain, pred_leaf=True)

        self.X_leaves = self.X_leaves.assign(**{f'leaf_{i}': pred_leaves[:, i] for i in range(self.trees_to_stop)})
        for i in range(self.trees_to_stop):
            self.filter_groups(dataset, i)

        indexes_to_filter = self.choose_groups(dataset)
        indexes_to_keep = sorted(set(X_real_train.index).difference(set(indexes_to_filter)))
        self.X_train_filtered, iter_dtrain = self.filter_df_and_dmatrix(iter_dtrain, X_real_train, indexes_to_keep)
        self.y_train_filtered = y_real_train[self.X_train_filtered.index]

        if self.prediction_model is None:
            if self.trees_number - self.trees_to_stop - 1 > 0:
                self.model = xgb.train(self.params, num_boost_round=self.trees_number - self.trees_to_stop,
                                       dtrain=iter_dtrain, xgb_model=self.model)
        else:
            self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)

        self.A = ahocorasick.Automaton()
        for key, target in self.hom_groups.items():
            self.A.add_word(key, (target, key))

        run_time = time.time() - s

        if self.prediction_model is None:
            preds = [round(p) for p in self.model.predict(xgb.DMatrix(data=dataset.X_test))]
        else:
            preds = self.prediction_model.predict(dataset.X_test)

        results = FilteringResults(score=dataset.metric(dataset.y_test, preds),
                                   run_time=run_time,
                                   filtered_score=dataset.metric(dataset.y_test, self.predict(dataset.X_test)),
                                   new_size_percent=len(self.X_train_filtered) / len(dataset.X_train))
        return results


class FilterDTSampleFinal(FilteringExperiment):

    def __init__(self, name,
                 trees_number=100, sample_percent=0, examples_to_keep=1000, prediction_model=None, trees_to_stop=None,
                 params=None,
                 index_name='index', threshold=40, n_jobs=24):
        super().__init__(name)
        self.model = None
        self.prediction_model = prediction_model if prediction_model is not None else None
        self.trees_number = trees_number
        self.sample_percent = sample_percent
        self.params = params
        self.trees_to_stop = trees_to_stop
        self.index_name = index_name
        self.X_leaves = None
        self.groups = None
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.examples_to_keep = examples_to_keep
        self.threshold = threshold
        self.results = []
        self.A = None
        self.X_train_filtered = None
        self.y_train_filtered = None
        self.original_sizes = None

    def reset_attributes(self) -> None:
        self.model = None
        self.X_leaves = None
        self.groups = None
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.results = []
        self.A = None
        self.X_train_filtered = None
        self.y_train_filtered = None

    def choose_groups(self, dataset: Dataset):
        sorted_candidates = sorted(self.hom_groups_candidates, key=lambda g: g.size, reverse=True)

        label_counter = defaultdict(int)
        label_amount = dict(dataset.y_train.value_counts())
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

    def predict(self, X_test):
        pred_leaves = self.model.apply(X_test)
        pred_leaves_df = pd.DataFrame(pred_leaves, index=X_test.index, columns=['leaf_id'])

        if self.prediction_model is None:
            preds = self.model.predict(X_test)
        else:
            preds = self.prediction_model.predict(X_test)
        pred_leaves_df.loc[:, 'pred_value'] = preds

        predictions = pred_leaves_df.apply(lambda row: self._predict_by_leafs(row['leaf_id'], row['pred_value']),
                                           axis=1)
        return predictions

    def get_predictions_df(self, X_test):
        pred_leaves = self.model.predict(xgb.DMatrix(data=(X_test)), pred_leaf=True)
        pred_leaves_df = pd.DataFrame(pred_leaves, index=X_test.index,
                                      columns=[f'leaf_{i}' for i in range(len(self.model.get_dump()))])

        if self.prediction_model is None:
            preds = self.model.predict(xgb.DMatrix(data=X_test))
        else:
            preds = self.prediction_model.predict(X_test)
        pred_leaves_df.loc[:, 'pred_value'] = preds

        pred_leaves_df.loc[:, 'joined'] = pred_leaves_df.apply(
            lambda row: '_'.join([str(row[c]) for c in pred_leaves_df.columns if c.startswith('leaf')]), axis=1)
        pred_leaves_df.loc[:, 'pred_by_groups'] = pred_leaves_df.apply(
            lambda row: self._is_in_homogeneous_group(row['joined']),
            axis=1)
        pred_leaves_df.loc[:, 'final_prediction'] = pred_leaves_df.apply(
            lambda row: self._predict_by_leafs(row['joined'], row['pred_value']), axis=1)
        return pred_leaves_df

    def filter_df_and_dmatrix(self, dmatrix: xgb.DMatrix, data: pd.DataFrame, indexes, index_name='index'):
        new_data = data.reset_index()
        filtered_data = new_data[new_data[index_name].isin(indexes)]
        slice_indexes = list(filtered_data.index)
        filtered_dmatrix = dmatrix.slice(slice_indexes)
        return filtered_data.set_index(index_name), filtered_dmatrix

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        X_real_train = dataset.X_train.sort_index()
        y_real_train = dataset.y_train.sort_index()
        self.original_sizes = dict(y_real_train.value_counts())
        self.X_leaves = pd.DataFrame(y_real_train)
        self.hom_groups = dict()
        self.groups = None
        if not self.params:
            self.model = DecisionTreeClassifier()
        else:
            self.model = DecisionTreeClassifier(**self.params)

        self.model.fit(X_real_train, y_real_train)
        pred_leaves = self.model.apply(X_real_train)
        self.X_leaves.loc[:, 'leaf_id'] = pred_leaves

        groups = self.X_leaves.groupby(['leaf_id']).agg(['sum', 'count'])[dataset.target_col].sort_values('count',
                                                                                                  ascending=False).reset_index()
        groups.loc[:, 'target_col'] = groups.apply(
            lambda row: int(row['sum'] / row['count']) if row['sum'] == 0 or row['sum'] == row['count'] else None,
            axis=1)
        groups.dropna(inplace=True)
        self.hom_groups_candidates = [FilteredGroup(key=row['leaf_id'], label=row['target_col'], size=row['count'],
                                                    group=self.X_leaves[
                                                        self.X_leaves['leaf_id'] == row['leaf_id']].index.tolist()) for
                                      index, row in groups.iterrows()]
        indexes_to_filter = self.choose_groups(dataset)
        indexes_to_keep = sorted(set(X_real_train.index).difference(set(indexes_to_filter)))
        self.X_train_filtered = X_real_train.loc[indexes_to_keep, :]
        self.y_train_filtered = y_real_train[self.X_train_filtered.index]

        if self.prediction_model is not None:
            self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)

        run_time = time.time() - s

        if self.prediction_model is None:
            preds = self.model.predict(dataset.X_test)
        else:
            preds = self.prediction_model.predict(dataset.X_test)

        results = FilteringResults(score=dataset.metric(dataset.y_test, preds),
                                   run_time=run_time,
                                   filtered_score=dataset.metric(dataset.y_test, self.predict(dataset.X_test)),
                                   new_size_percent=len(self.X_train_filtered) / len(dataset.X_train))
        return results


class CraigFilter(FilteringExperiment):

    def __init__(self, name, number_of_examples, model, metric='cosine', index_name='index', res=None):
        super().__init__(name)
        self.number_of_examples = number_of_examples
        self.model = model
        self.metric = metric
        self.index_name = index_name
        self.res = res

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        if self.res:
            sample_indexes, weights, _, _, _, _ = self.res
        else:
            sample_indexes, weights, _, _, _, _ = get_orders_and_weights(B=self.number_of_examples,
                                                                         X=StandardScaler().fit_transform(
                                                                             dataset.X_train),
                                                                         smtk=0,
                                                                         metric=self.metric,
                                                                         y=dataset.y_train.values)

        X_real_train = dataset.X_train.reset_index().loc[sample_indexes, :].set_index(self.index_name)
        y_real_train = dataset.y_train[X_real_train.index]
        self.model.fit(X_real_train, y_real_train, sample_weight=weights)

        run_time = time.time() - s
        score = dataset.metric(dataset.y_test, self.model.predict(dataset.X_test))
        results = FilteringResults(score=score,
                                   run_time=run_time,
                                   filtered_score=score,
                                   new_size_percent=len(X_real_train) / len(dataset.X_train))
        return results
