import pandas as pd
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sksurv.util import *
import numpy as np


data_x = pd.read_csv('train/features/radiomics.csv')
data_y = pd.read_csv('train/output.csv')
print(data_x.columns)
X = data_x[['shape.4', 'shape.6', 'shape.2', 'textural.9', 'textural.10', 'textural.11']].values[2:].astype(float)
y = np.stack([data_y['Event'].astype(bool).values, data_y['SurvivalTime'].astype(int).values], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = pd.DataFrame(X_train, columns=['shape.4', 'shape.6', 'shape.2', 'textural.9', 'textural.10', 'textural.11'])

y_train = Surv().from_arrays(y_train[:, 0], y_train[:, 1])
y_test = Surv().from_arrays(y_test[:, 0], y_test[:, 1])

estimator = CoxPHSurvivalAnalysis()
estimator.fit(X_train, y_train)

print(estimator.coef_)
print(estimator.score(X_test, y_test))
pred_surv = estimator.predict_survival_function(X_test)
for i, c in enumerate(pred_surv):
    plt.step(c.x, c.y, where="post", label="Sample %d" % (i + 1))
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.show()



