import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from select_k_best_grid_search import select_data


def make_confusion_matrix(estimator_info, x, y, cv):
    params = estimator_info['params']
    clf = MLPClassifier(**params)
    k_x = select_data(x, y, estimator_info['k'])
    y_pred = cross_val_predict(clf, x, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)
    print('confusion matrix:')
    print(conf_mat)
    plot_cm(y, y_pred)


def plot_cm(y_true, y_pred):
    skplt.metrics.plot_confusion_matrix(
        y_true,
        y_pred,
        figsize=(12, 12)
    )
    plt.savefig(f'../png/experiment/cm.png')
