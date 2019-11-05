import pandas as pd


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


def get_breast_cancer_data():
    breast_cancer = pd.read_csv('../data/breast-cancer-wisconsin.data')
    breast_cancer.columns = breast_cancer_columns
    breast_cancer.drop('id', axis=1, inplace=True)  # id do not play any role at classification
    breast_cancer = breast_cancer[breast_cancer['Bare Nuclei'] != '?']  # TODO take care of it (replace with mean or smth else)
    breast_cancer.dropna(inplace=True)
    breast_cancer['Bare Nuclei'] = breast_cancer['Bare Nuclei'].astype('int')
    return breast_cancer
