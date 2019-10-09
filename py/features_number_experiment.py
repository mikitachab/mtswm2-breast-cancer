import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning

import warnings
from tqdm import trange

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# project requirements constants
CV_REPEAT_NUMBER = 10  # 5
CV_FOLDS = 3  # 2
SCORE_METHOD = 'accuracy'
PARAMS_GRID = dict(
    hidden_layer_sizes=[(11,), (15,), (20,)],
    solver=['sgd', 'adam']  # 'adam' -> back propagation without momentum
)
CROSS_VALIDATION = StratifiedKFold(n_splits=CV_FOLDS)
ESTIMATOR = MLPClassifier()


def make_train_test_split(data, target_column):
    x = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return dict(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test
    )


def select_k_best(data, k):
    x = data['x_train']
    y = data['y_train']
    k_best_selector = SelectKBest(score_func=f_classif, k=k)
    selected_data = {}
    selected_data['x_train'] = k_best_selector.fit_transform(x, y)
    selected_data['x_test'] = k_best_selector.transform(data['x_test'])
    return {**data, **selected_data}


def get_params_score(cv_results):
    return pd.DataFrame(cv_results)[['params', 'mean_test_score']].to_dict()


def run_experiment(breast_cancer_data):
    grid_search = GridSearchCV(
        param_grid=PARAMS_GRID,
        estimator=ESTIMATOR,
        cv=CROSS_VALIDATION
    )

    features_scores = []
    params_scores = {}

    learning_data = make_train_test_split(data=breast_cancer_data, target_column='Class')
    features_numbers = learning_data['x_train'].shape[1]

    print(f'Running experiment for {features_numbers} features')

    for k in trange(1, features_numbers + 1):
        k_features_score = []
        for _ in range(CV_REPEAT_NUMBER):
            selected_data = select_k_best(learning_data, k)
            grid_search.fit(selected_data['x_train'], selected_data['y_train'])
            params_scores[k] = get_params_score(grid_search.cv_results_)
            k_features_score.append(grid_search.score(selected_data['x_test'], selected_data['y_test']))
        features_scores.append(np.mean(k_features_score))

    return features_scores, params_scores
