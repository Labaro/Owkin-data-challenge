import pandas as pd
import numpy as np


def get_data(r, cl):
    """Read radiomics, clinical and output csv files and returns data after being homogenize.

    :parameter str r: path to radiomics file
    :parameter str cl: path to clinical data file

    :return pd.Dataframe X: samples features.
    :return pd.DataFrame y: output.
    """
    radiomics = pd.read_csv(r)
    df_radio = pd.DataFrame(radiomics.values[2:, 1:].astype(float),
                            columns=radiomics.values[0, 1:])

    clinical = pd.read_csv(cl)
    clinical_array = clinical.values

    # Homogenization of Histology feature:
    hist = np.apply_along_axis(lambda x: np.char.lower(x), 0, clinical_array[:, 1].astype(str))
    hist[np.flatnonzero(np.core.defchararray.find(hist, 'nos') != -1)] = 'nos'  # find every elements containing 'nos'
    for name in ['adenocarcinoma', 'large cell', 'squamous cell carcinoma', 'nos', 'nan']:
        clinical[name] = (hist == name).astype(int)
    del clinical['Histology']

    del clinical['age']

    # Homogenization of SourceDataset feature:
    source = clinical_array[:, 4]
    for name in ['l1', 'l2']:
        clinical[name] = (source == name).astype(int)
    del clinical['SourceDataset']

    # Concatenation with radiomics file:
    X = pd.DataFrame(np.hstack([clinical.values, df_radio.values]),
                     columns=np.concatenate((clinical.columns, df_radio.columns))).set_index('PatientID')

    # Homogenization of output file:
    y = pd.read_csv('train/output.csv')
    y['Event'] = y['Event'] == 1
    del y['PatientID']

    return X, y


def separate_data(X, y, separator=None):
    """Separate the data depending on histology or dataset source.
    :parameter pd.DataFrame X: features set
    :parameter pd.DataFrame y: prediction set
    :parameter separator: optional, str or None (default=None).
                            - 'Histology' > return one DataFrame per different histology
                            - 'SourceDataset' > return one Dataframe per different dataset source

    :returns X, y, idxs: list of pd.Dataframe, pd.DataFrame and array-like. The indexes identifying the position of the
    sample in the original set X."""
    Xs, ys, idxs = [], [], []
    if separator is None:
        return [X], [y]
    elif separator == 'Histology':
        names = ['adenocarcinoma', 'large cell', 'squamous cell carcinoma', 'nos', 'nan']
        for name in names:
            index = np.flatnonzero(X[name])
            Xs.append(X.iloc[index].drop(names, axis=1))
            ys.append(y.iloc[index])
            idxs.append(index)
    elif separator == 'SourceDataset':
        names = ['l1', 'l2']
        for name in names:
            index = np.flatnonzero(X[name])
            Xs.append(X.iloc[index].drop(names, axis=1))
            ys.append(y.iloc[index])
            idxs.append(index)
    return Xs, ys, idxs


def to_pd(index, index_test, T_pred):
    """Transform the prediction into a proper Dataframe object to be used in cindex
    :parameter list index: PatientID
    :parameter list index_test: list of list. Contains index of position into the original file.
    :parameter list T_pred: list of list. Contains the predicted survival time."""
    N = len(index)
    T, I = np.zeros(N), np.zeros(N, dtype=int)
    for i, idx in enumerate(index_test):
        I[idx] = [index[j] for j in index_test[i]]
        T[idx] = T_pred[i]
    idx = np.argwhere(T == 0)
    T = np.delete(T, idx)
    I = np.delete(I, idx)
    empty = np.empty(N - len(idx))
    empty[:] = np.NAN
    y_pred = pd.DataFrame({'PatientID': I, 'SurvivalTime': T, 'Event': empty}).set_index('PatientID')
    return y_pred


def LeverageScoresSampler(A, k, theta):
    """Deterministic column sampling
    :parameter array-like A: matrix of features that need to reduced.
    :parameter int k: parameter to choose the k first singular vector ordered in non increasing order
    :parameter flot theta: energy of the final reduction. Close to k is best."""
    u, s, vh = np.linalg.svd(A)
    Vk = vh.T[:k + 1]
    l = np.apply_along_axis(lambda x: np.linalg.norm(x), 0, Vk)
    idx = np.argsort(l)[::-1]
    l = l[idx]
    L, c = 0, 0
    for c in range(A.shape[1] - 1):
        L += l[c]
        if L > theta:
            break
    if c < k:
        c = k
    S = np.zeros((A.shape[1], c))
    for i in range(c):
        S[idx[i], i] = 1
    return S, idx[:c + 1]
