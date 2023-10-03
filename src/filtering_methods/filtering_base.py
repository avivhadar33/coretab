from src.dataset import Dataset
from abc import ABC
from dataclasses import dataclass

from sklearn.model_selection import KFold


@dataclass
class FilteringResults:
    score: int
    filtered_score: int
    run_time: int
    new_size_percent: int


class FilteringExperiment(ABC):

    def __init__(self, name):

        self.name = name

        self.scores = []
        self.scores_filtered = []
        self.run_times = []
        self.new_size_percents = []

        self.trials_number = None

        self.save_each_iter = None
        self.iter_results = None

        self.trials_number = None

    def sample_func(self, dataset: Dataset, p) -> FilteringResults:
        pass

    def reset_attributes(self) -> None:
        pass

    def test_filter_method(self,
                           dataset: Dataset,
                           trials_number=1,
                           print_results=False,
                           save_each_iter=False):

        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        for _ in range(self.trials_number):
            results = self.sample_func(dataset, save_each_iter)
            if print_results:
                print(results)
            self.scores.append(results.score)
            self.run_times.append(results.run_time)
            self.scores_filtered.append(results.filtered_score)
            self.new_size_percents.append(results.new_size_percent)

    def test_method_cv(self,
                           dataset: Dataset,
                           trials_number=1,
                           cv=4,
                           print_results=False,
                           save_each_iter=False,
                           random_seed = 42):

        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        for repeat in range(trials_number):
            # Initialize a list to store fold scores for this repeat
            fold_scores = []

            # Create a k-fold cross-validation object for this repeat
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed + repeat)

            # Inner loop for folds
            for train_index, test_index in kf.split(dataset.df):
                self.reset_attributes()
                X, y = dataset.df.drop(dataset.target_col, axis=1), dataset.df[dataset.target_col]
                X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                trial_dataset = Dataset(df=dataset.df, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                        target_col=dataset.target_col, metric=dataset.metric)
                results = self.sample_func(trial_dataset, save_each_iter)
                if print_results:
                    print(results)
                self.scores.append(results.score)
                self.run_times.append(results.run_time)
                self.scores_filtered.append(results.filtered_score)
                self.new_size_percents.append(results.new_size_percent)







