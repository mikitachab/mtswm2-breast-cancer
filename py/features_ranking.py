from sklearn.feature_selection import SelectKBest, f_classif


def get_features_ranking(x, y):
    n_features = x.shape[1]
    # f_classif: ANOVA test (F-value between label/feature for regression tasks)
    k_best_selector = SelectKBest(score_func=f_classif, k=n_features)
    k_best_selector.fit(x, y)
    scores = k_best_selector.scores_
    column_scores = [
        {'name': name, 'score': round(score, 2)}
        for name, score in zip(x.columns, scores)
    ]
    return sorted(column_scores, key=lambda x: x['score'], reverse=True)
