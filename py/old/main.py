import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pprint import pprint

from features_ranking import get_features_ranking
from features_number_experiment import run_experiment


breast_cancer_columns = [
    'id',
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
    'Class'
]


def main():
    breast_cancer_data = get_breast_cancer_data()
    x = breast_cancer_data.drop('Class', axis=1)
    y = breast_cancer_data['Class']

    # features ranking
    print('make features ranking')
    features_ranking = get_features_ranking(x, y)
    print_ranking(features_ranking)
    save_json('features_ranking.json', features_ranking)

    # experiment
    features_scores, params_scores = run_experiment(breast_cancer_data)

    save_json('features_scores.json', features_scores)
    save_json('params_scores.json', params_scores)

    print('features scores:')
    pprint(features_scores)

    print('params scores')
    pprint(params_scores)

    plot_result('../png/result.png', features_scores)


def get_breast_cancer_data():
    breast_cancer = pd.read_csv('../data/breast-cancer-wisconsin.data')
    breast_cancer.columns = breast_cancer_columns
    breast_cancer.drop('id', axis=1, inplace=True)  # id do not play any role at classification
    breast_cancer = breast_cancer[breast_cancer['Bare Nuclei'] != '?']  # TODO take care of it (replace with mean or smth else)
    return breast_cancer


def print_ranking(features_ranking):
    print('Breast Cancer Dataset Features Ranking (ANOVA Test)')
    for index, feature in enumerate(features_ranking, 1):
        print(f"{index}. {feature['name']} {feature['score']}")


def save_json(json_name, data):
    with open(json_name, 'w') as file:
        file.write(json.dumps(data, indent=2))


def plot_result(plot_filename, features_scores):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=pd.DataFrame(dict(
            k_features=list(range(1, len(features_scores) + 1)),
            scores=features_scores
        )),
        x='k_features',
        y='scores'
    )
    plt.savefig(plot_filename)


if __name__ == '__main__':
    main()
