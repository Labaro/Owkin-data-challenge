from metrics_t9gbvr2 import *
from sklearn.base import clone


def scorer(y_truth, y_pred, index):
    """Return the c-index value of a prediction.
    y_truth: DataFrame
    y_pred: array"""
    data = pd.read_csv("train/output.csv", index_col=0)
    d_truth = {'SurvivalTime': y_truth.tolist(), 'Event': data['Event']}
    df_truth = pd.DataFrame(d_truth, index=index)
    d_pred = {'SurvivalTime': y_pred.tolist(), 'Event': np.zeros(y_pred.shape[0])}
    df_pred = pd.DataFrame(d_pred, index=index)
    return cindex(df_truth, df_pred)


def get_random_train_test(X, y, n, index):
    idx = np.random.randint(0, y.size, n)
    idx_C = np.setdiff1d(np.arange(y.size), idx)
    X_train, y_train = X[idx], y[idx]
    X_test, y_test = X[idx_C], y[idx_C]
    index_train = index[idx]
    index_test = index[idx_C]
    return X_train, X_test, y_train, y_test, index_train, index_test


def get_train_test(X, y, a, b, index):
    idx = np.arange(a, b)
    idx_C = np.setdiff1d(np.arange(y.size), idx)
    X_train, y_train = X[idx_C], y[idx_C]
    X_test, y_test = X[idx], y[idx]
    index_train = index[idx_C]
    index_test = index[idx]
    return X_train, X_test, y_train, y_test, index_train, index_test


def cross_validation(model, X, y, index, n):
    N = X.shape[0]
    S = 0
    for i in range(n):
        X_train, X_test, y_train, y_test, index_train, index_test = get_train_test(X, y, (N // n) * i, (N//n) * (i + 1),
                                                                                   index)
        mdl = clone(model)
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        S += scorer(y_test, y_pred, index_test)
    return S/n


def cross_validation_average(models, X, y, index, n):
    N = X.shape[0]
    S = 0
    for i in range(n):
        X_train, X_test, y_train, y_test, index_train, index_test = get_train_test(X, y, (N // n) * i, (N//n) * (i + 1),
                                                                                   index)
        pred = []
        for model in models:
            mdl = clone(model)
            mdl.fit(X_train, y_train)
            pred.append(mdl.predict(X_test))
        S += scorer(y_test, np.average(pred, axis=0, weights=[0.2, 0.8]), index_test)
    return S / n