import json
from collections import OrderedDict
import warnings

import pandas
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
from tabulate import tabulate

from utils import product_from_dict
import numpy as np


warnings.filterwarnings('ignore')


class SelectKBestGridSearch:
    def __init__(self, param_grid, estimator_cls, cv):
        self.param_grid = param_grid
        self.cv = cv
        self.estimator_cls = estimator_cls
        self.estimator = None
        self.result = {}
        self.result_table = {}
        self.best_estimator_info = {}
        self.max_score = np.NINF

    def _cv_eval(self, estimator, x, y):
        """fit estiamtor with given cv object """
        if type(y) == pandas.Series:
            y = y.values
        scores = []
        for train_index, test_index in self.cv.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator.fit(X_train, y_train)
            score = estimator.score(X_test, y_test)
            scores.append(score)
        return np.mean(scores)

    def _fit_params_set(self, params, x, y):
        """fit estiamator with given params for every k features"""
        k_scores = OrderedDict()
        n_features = x.shape[1]
        estimator = self.estimator_cls(**params)
        for k in tqdm(range(1, n_features + 1), ncols=50):
            selected_x = select_data(x, y, k)
            score = self._cv_eval(estimator, selected_x, y)

            if score > self.max_score:
                self.best_estimator_info = {
                    'score': score,
                    'params': params,
                    'k': k,
                    'estimator_name': self.estimator_cls.__name__
                }
                self.max_score = score

            k_scores[k] = score
        return k_scores

    def fit(self, x, y):
        """fit estimator with given estimatir_cls with every parametres combinations for every features number"""
        params_combinations = product_from_dict(self.param_grid)

        estimator_name = self.estimator_cls.__name__
        n_features = x.shape[1]
        params_len = len(params_combinations)
        print(f'run {n_features} fits of {estimator_name} for {params_len} parametres sets, total {params_len*n_features} fits')

        for i, params in enumerate(product_from_dict(self.param_grid), 1):
            print(f'run {n_features} fits for', params, f'{i}/{params_len}')
            params_key = json.dumps(params)  # convert dict to str because dict cant be a dict key
            self.result[params_key] = self._fit_params_set(params, x, y)
            self._print_params_result(params_key)

    def _print_params_result(self, params_key):
        print(params_key, 'results:')
        result_table = tabulate(
            tabular_data=self.result[params_key].items(),
            headers=['k', 'score'],
            tablefmt='github'
        )
        self.result_table[params_key] = result_table
        print(result_table)

    def get_result(self, table=False):
        return self.result if not table else self.result_table

    def get_max_score(self):
        return self.max_score

    def get_best_estimator_info(self):
        return self.best_estimator_info


def select_data(x, y, k):
    k_best_selector = SelectKBest(score_func=f_classif, k=k)
    return k_best_selector.fit_transform(x, y)
