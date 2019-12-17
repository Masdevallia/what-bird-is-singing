
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

featuresDf = pd.read_pickle('./dataset/featuresDF_1.pkl')

#.......................................................................................

# Fourier+MFCC:

featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                              featuresDf.mfcc[i]]) for i in range(len(featuresDf))]

X = np.array(featuresDf['fourier_mfcc'].tolist())
y = np.array(featuresDf['class'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2)

# Model testing:
models = {'LogisticRegression': LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=1000),
          'SVC': SVC(gamma='auto'),
          'KNeighborsClassifier': KNeighborsClassifier(3),
          'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
          'DecisionTreeClassifier': DecisionTreeClassifier(),
          'GradientBoostingClassifier': GradientBoostingClassifier()}

metrics = {}
for modelName, model in models.items():
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics[modelName] = {'accuracy': round(accuracy_score(y_test, y_pred),2),
                          'precision': round(precision_score(y_test, y_pred, average='weighted'),2),
                          'recall': round(recall_score(y_test, y_pred, average='weighted'),2)}


#{'LogisticRegression': {'accuracy': 0.55, 'precision': 0.52, 'recall': 0.55},
# 'SVC': {'accuracy': 0.5, 'precision': 0.25, 'recall': 0.5},
# 'KNeighborsClassifier': {'accuracy': 0.6, 'precision': 0.61, 'recall': 0.6},
# 'RandomForestClassifier': {'accuracy': 0.88, 'precision': 0.89, 'recall': 0.88},
# 'DecisionTreeClassifier': {'accuracy': 0.78, 'precision': 0.79, 'recall': 0.78},
# 'GradientBoostingClassifier': {'accuracy': 0.92, 'precision': 0.92, 'recall': 0.92}}

#.......................................................................................

# MFCC:

X = np.array(featuresDf['mfcc'].tolist())

#{'LogisticRegression': {'accuracy': 0.78, 'precision': 0.78, 'recall': 0.78},
# 'SVC': {'accuracy': 0.58, 'precision': 0.77, 'recall': 0.58},
# 'KNeighborsClassifier': {'accuracy': 0.93, 'precision': 0.93, 'recall': 0.93},
# 'RandomForestClassifier': {'accuracy': 0.91, 'precision': 0.92, 'recall': 0.91},
# 'DecisionTreeClassifier': {'accuracy': 0.78, 'precision': 0.78, 'recall': 0.78},
# 'GradientBoostingClassifier': {'accuracy': 0.92, 'precision': 0.92, 'recall': 0.92}}

#.......................................................................................

# Fourier:

X = np.array(featuresDf['fourier'].tolist())

#{'LogisticRegression': {'accuracy': 0.55, 'precision': 0.52, 'recall': 0.55},
# 'SVC': {'accuracy': 0.5, 'precision': 0.25, 'recall': 0.5},
# 'KNeighborsClassifier': {'accuracy': 0.6, 'precision': 0.6, 'recall': 0.6},
# 'RandomForestClassifier': {'accuracy': 0.65, 'precision': 0.66, 'recall': 0.65},
# 'DecisionTreeClassifier': {'accuracy': 0.53, 'precision': 0.53, 'recall': 0.53},
# 'GradientBoostingClassifier': {'accuracy': 0.65, 'precision': 0.64, 'recall': 0.65}}

#.......................................................................................

# GradientBoostingClassifier GridSearchCV:
parameters = { 
    'n_estimators': [300, 500, 700],
    'learning_rate' :[0.3, 0.5]}
gbc = GradientBoostingClassifier() 
clf = GridSearchCV(gbc, parameters, cv=5, scoring='accuracy', verbose=5, n_jobs= -1)
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.best_score_) 
print(clf.best_params_)
# GridSearchCV refits an estimator using the best found parameters on the whole dataset.

# GradientBoostingClassifier(criterion='friedman_mse', init=None,
                           # learning_rate=0.1, loss='deviance', max_depth=3,
                           # max_features=None, max_leaf_nodes=None,
                           # min_impurity_decrease=0.0, min_impurity_split=None,
                           # min_samples_leaf=1, min_samples_split=2,
                           # min_weight_fraction_leaf=0.0, n_estimators=400,
                           # n_iter_no_change=None, presort='auto',
                           # random_state=None, subsample=1.0, tol=0.0001,
                           # validation_fraction=0.1, verbose=0,
                           # warm_start=False)
# 0.7905631383892253
# {'learning_rate': 0.1, 'n_estimators': 400}

#.......................................................................................

import h2o
from h2o.automl import H2OAutoML
h2o.init(nthreads = -1, max_mem_size = 6)
DF = pd.DataFrame(X, columns=[f'fm{e}' for e in range(1,385)])
DF['class'] = y
hf = h2o.H2OFrame(DF)
y_columns = 'class'
# Fitting models (AutoML):
aml_ti = H2OAutoML(max_models= 10, seed= 1, nfolds=5, sort_metric='RMSE')
aml_ti.train(y = y_columns, training_frame = hf) # x = x_columns
lb_ti = aml_ti.leaderboard
print(aml_ti.leader)
# pred = aml_ti.leader.predict()
# save the model
model_path = h2o.save_model(model=aml_ti, path="./models", force=True)
# load the model
# saved_model = h2o.load_model('./models')
h2o.cluster().shutdown()

#.......................................................................................

# https://stackabuse.com/scikit-learn-save-and-restore-models/
from sklearn.externals import joblib
# Save to file
# joblib_file = "GradientBoostingClassifier.pkl"
joblib_file = "./models/GradientBoostingClassifier.pkl"
joblib.dump(clf, joblib_file)
# Load from file
# joblib_model = joblib.load(joblib_file)
joblib_model = joblib.load("./models/GradientBoostingClassifier.pkl")
# Calculate the accuracy and make predictions
score = joblib_model.score(X_test, y_test)
Ypredict = joblib_model.predict(X_test)

#.......................................................................................

# When a new audio is received, the app cuts it into 'n' windows:
testing = featuresDf[featuresDf['id']==168257]
# Then makes the prediction using the chosen model on all windows:
topredict = np.array(testing['fourier_mfcc'].tolist())
prediction = clf.predict(topredict)
# Then counts how many times each species appears in the predictions:
from collections import Counter
windowPredictions = Counter(prediction)
# Finally, it returns the species that appears more times:
max_key = max(windowPredictions, key=lambda x: windowPredictions[x])
print(max_key)

