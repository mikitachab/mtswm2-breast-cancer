import json
from collections import OrderedDict

import pandas
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import trange

from utils import product_from_dict

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class SelectKBestGridSearch:
    def __init__(self, param_grid, estimator_cls, cv):
        self.param_grid = param_grid
        self.cv = cv
        self.estimator_cls = estimator_cls
        self.estimator = None
        self.result = {}

    def _cv_eval(self, x, y):
        if type(y) == pandas.Series:
            y = y.values
        scores = []
        for train_index, test_index in self.cv.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.estimator.fit(X_train, y_train)
            scores.append(self.estimator.score(X_test, y_test))
        return np.mean(scores)

    def _get_params_result(self, x, y):
        k_scores = OrderedDict()
        n_features = x.shape[1]
        for k in trange(1, n_features + 1):
            selected_x = select_data(x, y, k)
            score = self._cv_eval(selected_x, y)
            k_scores[k] = score
        return k_scores

    def fit(self, x, y):
        params_combinations = product_from_dict(self.param_grid)
        print(f'run {x.shape[1]} fits of {self.estimator_cls.__name__} for {len(params_combinations)} parametres')

        for params in product_from_dict(self.param_grid):
            self.estimator = self.estimator_cls(**params)
            params_key = json.dumps(params)  # convert dict to str because dict cant be a dict key
            self.result[params_key] = self._get_params_result(x, y)
            self._print_params_result(params_key)

    def _print_params_result(self, params_key):
        print(params_key)
        for k, score in self.result[params_key].items():
            print('k:', k, 'score:', score)

    def get_result(self):
        return self.result


def select_data(x, y, k):
    k_best_selector = SelectKBest(score_func=f_classif, k=k)
    return k_best_selector.fit_transform(x, y)
