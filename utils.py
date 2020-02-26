import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sksurv.util import Surv


def get_data():
    """Read radiomics, clinical and output csv files and returns data after being homogenize.

    :return pd.Dataframe X: samples features.
    :return pd.DataFrame y: output.
    """
    radiomics = pd.read_csv('train/features/radiomics.csv')
    df_radio = pd.DataFrame(radiomics.values[2:, 1:].astype(float),
                            columns=radiomics.values[0, 1:])

    clinical = pd.read_csv('train/features/clinical_data.csv')
    clinical_array = clinical.values

    # Homogenization of Histology feature:
    hist = np.apply_along_axis(lambda x: np.char.lower(x), 0, clinical_array[:, 1].astype(str))
    hist[np.flatnonzero(np.core.defchararray.find(hist, 'nos') != -1)] = 'nos'  # find every elements containing 'nos'
    for name in ['adenocarcinoma', 'large cell', 'squamous cell carcinoma', 'nos', 'nan']:
        clinical[name] = (hist == name).astype(int)
    del clinical['Histology']

    # Homogenization of SourceDataset feature:
    source = clinical_array[:, 4]
    for name in ['l1', 'l2']:
        clinical[name] = (source == name).astype(int)
    del clinical['SourceDataset']

    del clinical['PatientID']

    # Concatenation with radiomics file:
    X = pd.DataFrame(np.hstack([clinical.values, df_radio.values]),
                     columns=np.concatenate((clinical.columns, df_radio.columns)))

    # Homogenization of output file:
    y = pd.read_csv('train/output.csv')
    y['Event'] = y['Event'] == 1
    del y['PatientID']

    return X, y


def separate_data(X, y, separator=None):
    """Separate the data depending on histology or dataset source.
    :parameter separator: optional, str or None (default=None).
                            - 'Histology' > return one DataFrame per different histology
                            - 'SourceDataset' > return one Dataframe per different dataset source
    :returns X, y: list of """
    Xs, ys = [], []
    if separator is None:
        return [X], [y]
    elif separator == 'Histology':
        names = ['adenocarcinoma', 'large cell', 'squamous cell carcinoma', 'nos', 'nan']
        for name in names:
            index = np.flatnonzero(X[name])
            Xs.append(X.iloc[index].drop(names, axis=1))
            ys.append(y.iloc[index])
    elif separator == 'SourceDataset':
        names = ['l1', 'l2']
        for name in names:
            index = np.flatnonzero(X[name])
            Xs.append(X.iloc[index].drop(names, axis=1))
            ys.append(y.iloc[index])
    return Xs, ys


def select_features(X, features=None):
    output = []
    if features is None:
        output = X
    else:
        for x in X:
            x.drop(np.setdiff1d(x.columns, features), axis=1)
            output.append(x)
    return output


def encode_data(Xs, ys, normalized=False):
    output_X, output_y = [], []
    for x in Xs:
        if normalized:
            scaler = preprocessing.StandardScaler()
            output_X.append(scaler.fit_transform(x.values))
        else:
            output_X.append(x.values)
    for y in ys:
        output_y.append(Surv().from_arrays(y.values[:, 1], y.values[:, 0]))
    return output_X, output_y


def split(features, test_size=None, separator=None, normalized=False):
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    Xs_train, ys_train = separate_data(X_train, y_train, separator=separator)
    Xs_test, ys_test = separate_data(X_test, y_test, separator=separator)
    Xs_train, ys_train = encode_data(select_features(Xs_train, features=features), ys_train, normalized=normalized)
    Xs_test, ys_test = encode_data(select_features(Xs_test, features=features), ys_test, normalized=normalized)
    return Xs_train, Xs_test, ys_train, ys_test
