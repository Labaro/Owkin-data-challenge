from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from utils import *
import matplotlib.pyplot as plt
from sklearn import preprocessing


def plotAutocovariance(data):
    ACov = np.cov(data, rowvar=False, bias=False)
    print('Covariance matrix:\n', ACov)

    cmap = sns.color_palette("GnBu", 40)
    sns.heatmap(ACov, cmap=cmap, vmin=0)
    plt.show()
    return ACov


train_output = pd.read_csv('train/output.csv', index_col=0)
train_input = pd.read_csv('train/features/radiomics.csv', index_col=0)
clinical = pd.read_csv('train/features/clinical_data.csv', index_col=0)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(train_input.values[2:].astype(float))
y = train_output['SurvivalTime'].values.astype(float)
index = train_output.index
Z = clinical[['Mstage', 'Nstage', 'Tstage']].values.astype(float)

A = np.cov(np.hstack((X, y.reshape((-1, 1)))), rowvar=False)
b = np.linalg.norm(A, ord=1, axis=0)
arg = np.argsort(A[-1, :-1])[:10]
print(train_input.values[0, arg])


X_train, X_test, y_train, y_test, idx_train, idx_test = get_random_train_test(np.hstack((X[:, arg], Z)), y, 340, index)


rf = RandomForestRegressor(n_estimators=100, criterion='mse', oob_score=True)
knn = KNeighborsRegressor(n_neighbors=5)


print("Random Forest: ", cross_validation(rf, X, y, index, 6))
print("K-nn: ", cross_validation(knn, X, y, index, 6))
print("Average of both: ", cross_validation_average([knn, rf], X, y, index, 6))
