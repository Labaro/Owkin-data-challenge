from pysurvival.models.semi_parametric import CoxPHModel

from utils import *

# In each variables the 1 and 2 are distinguishing the source center. 1 for l1 and 2 for l2.

output = pd.read_csv('train/output.csv')

data_train, _ = get_data('train/features/radiomics.csv', 'train/features/clinical_data.csv')
data_test, _ = get_data('test/features/radiomics.csv', 'test/features/clinical_data.csv')

index_train = list(data_train.index)
index_test = list(data_test.index)

Xs, ys, idxs = separate_data(data_train, output, separator='SourceDataset')
X_train1, X_train2 = Xs[0], Xs[1]
y_train1, y_train2 = ys[0], ys[1]
idx_train1, idx_train2 = idxs[0], idxs[1]

Xs_test, ys_test, idxs_test = separate_data(data_test, output, separator='SourceDataset')
X_test1, X_test2 = Xs_test[0], Xs_test[1]
idx_test1, idx_test2 = idxs_test[0], idxs_test[1]

S1, idx1 = LeverageScoresSampler(X_train1, 7, 6.9)
S2, idx2 = LeverageScoresSampler(X_train2, 7, 6.9)

print("Features sampling result:", X_train1.columns[idx1], X_train1.columns[idx2])

X_train1, X_test1 = np.dot(X_train1, S1), np.dot(X_test1, S1)
X_train2, X_test2 = np.dot(X_train2, S2), np.dot(X_test2, S2)

T_train1, T_train2 = y_train1['SurvivalTime'].loc[idx_train1].reset_index(drop=True), \
                     y_train2['SurvivalTime'].loc[idx_train2].reset_index(drop=True)
E_train1, E_train2 = y_train1['Event'].loc[idx_train1].reset_index(drop=True), \
                     y_train2['Event'].loc[idx_train2].reset_index(drop=True)

coxph1 = CoxPHModel()
coxph2 = CoxPHModel()
coxph1.fit(X_train1, T_train1, E_train1, lr=5e-2, l2_reg=1e-4, init_method='glorot_normal', max_iter=1000, verbose=True)
coxph2.fit(X_train2, T_train2, E_train2, lr=5e-2, l2_reg=1e-4, init_method='glorot_normal', max_iter=1000, verbose=True)

sur_func1 = coxph1.predict_survival(X_test1)
sur_func2 = coxph2.predict_survival(X_test2)

T1 = []
for f in sur_func1:
    t = 0
    for i, b in enumerate(coxph1.time_buckets):
        t += (b[1] - b[0]) * f[i + 1]
    T1.append(t + 1.01)

T2 = []
for f in sur_func2:
    t = 0
    for i, b in enumerate(coxph2.time_buckets):
        t += (b[1] - b[0]) * f[i + 1]
    T2.append(t + 1.01)

df = to_pd(index_test, [idx_test1, idx_test2], [T1, T2])
df.to_csv("out-notsubmitted-gl-sameS.csv", na_rep='nan')