import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold

from select_k_best_grid_search import SelectKBestGridSearch
from sklearn.neural_network import MLPClassifier
from dataset import get_breast_cancer_data

N_REPEATS = 5
N_SPLITS = 2


def main():
    params_grid = dict(
        hidden_layer_sizes=[(11,), (15,), (20,)],
        solver=['sgd', 'adam']  # 'adam' -> back propagation without momentum
    )
    breast_cancer = get_breast_cancer_data()
    x = breast_cancer.drop('Class', axis=1)
    y = breast_cancer['Class']
    rskf_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=N_REPEATS, random_state=42)
    select_k_best_gs = SelectKBestGridSearch(
        param_grid=params_grid,
        cv=rskf_cv,
        estimator_cls=MLPClassifier
    )
    select_k_best_gs.fit(x, y)
    gs_result = select_k_best_gs.get_result()

    for params_key, result in gs_result.items():
        plot_params_result(result=result, title=params_key)


def get_filename_from_mlp_params_key(params_key):
    params = json.loads(params_key)
    return f'hidden_layer_sizes_{params["hidden_layer_sizes"][0]}_solver_{params["solver"]}'


def plot_params_result(result, title):
    ks = list(result.keys())
    scores = list(result.values())
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=pd.DataFrame(dict(
            k_features=ks,
            scores=scores
        )),
        x='k_features',
        y='scores'
    )
    plt.title(title)
    imgfilename = get_filename_from_mlp_params_key(title)
    plt.savefig(f'../png/experiment/{imgfilename}.png')


if __name__ == '__main__':
    main()
