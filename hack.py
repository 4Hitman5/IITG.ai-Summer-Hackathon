import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc1 = StandardScaler()
sc2 = MinMaxScaler()

X = train.drop(columns=['Target', 'Id'], axis=1)
X = np.array(sc2.fit_transform(X))
y = np.array(train['Target'])

X_test = test.drop(columns=['Id'], axis=1)
X_test = np.array(sc2.transform(X_test))

pca = PCA(n_components=20)
pca.fit_transform(X)
pca.fit_transform(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
# classifier = LogisticRegression(tol=1e-7, max_iter=500, solver='liblinear')
# classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                         max_depth=10, max_features='auto', max_leaf_nodes=None,
#                         min_impurity_decrease=1e-07, min_samples_leaf=1,
#                         min_samples_split=2, min_weight_fraction_leaf=0.0,
#                         n_estimators=800, n_jobs=1, oob_score=False, random_state=None,
#                         verbose=0, warm_start=False)
# classifier = tree.DecisionTreeClassifier(max_depth=20)
classifier = GradientBoostingClassifier(n_estimators=300, max_depth=3)
classifier.fit(X_train, y_train)
prob1 = classifier.predict_proba(X_train)
prob2 = classifier.predict_proba(X_valid)
print(roc_auc_score(y_train, prob1[:, 1]))
print(roc_auc_score(y_valid, prob2[:, 1]))

final_pred = classifier.predict_proba(X_test)
dict = {
	'Id': test['Id'],
	'Target': final_pred[:, 1]
}
df = pd.DataFrame(dict)
df.set_index('Id', inplace=True)
df.to_csv('submission_final.csv')
