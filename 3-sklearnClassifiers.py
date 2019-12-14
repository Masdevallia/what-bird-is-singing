
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

featuresDf = pd.read_pickle('./dataset/featuresDF.pkl')

#.......................................................................................                              

# Checking if the dataframe is balanced (more or less the same number of samples in each class):
# for e in set(featuresDf['class']):
    # print(e, len(featuresDf[featuresDf['class'] == e]))
# Fixing it:
featuresDfBalanced = featuresDf.groupby('class')
featuresDfBalanced = pd.DataFrame(featuresDfBalanced.apply(
                     lambda x: x.sample(featuresDfBalanced.size().min()).reset_index(drop=True)))

#.......................................................................................

# Fourier+MFCC:

featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                              featuresDf.mfcc[i]]) for i in range(len(featuresDf))]
                     
X = np.array(featuresDf['fourier_mfcc'].tolist())
y = np.array(featuresDf['class'].tolist())
# X = np.array(featuresDfBalanced['fourier_mfcc'].tolist())
# y = np.array(featuresDfBalanced['class'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2)

# Model testing:
models = {'LogisticRegression': LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=1000),
          'SVC': SVC(gamma='auto'),
          'KNeighborsClassifier': KNeighborsClassifier(3),
          'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
          'DecisionTreeClassifier': DecisionTreeClassifier(),
          'GradientBoostingClassifier': GradientBoostingClassifier()}

# Scaled:
metrics = {}
for modelName, model in models.items():
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics[modelName] = {'accuracy': round(accuracy_score(y_test, y_pred),2),
                          'precision': round(precision_score(y_test, y_pred, average='weighted'),2),
                          'recall': round(recall_score(y_test, y_pred, average='weighted'),2)}

# Without balancing data:
#{'LogisticRegression': {'accuracy': 0.57, 'precision': 0.54, 'recall': 0.57},
# 'SVC': {'accuracy': 0.52, 'precision': 0.49, 'recall': 0.52},
# 'KNeighborsClassifier': {'accuracy': 0.6, 'precision': 0.61, 'recall': 0.6},
# 'RandomForestClassifier': {'accuracy': 0.88,'precision': 0.9,'recall': 0.88},
# 'DecisionTreeClassifier': {'accuracy': 0.78,'precision': 0.78,'recall': 0.78},
# 'GradientBoostingClassifier': {'accuracy': 0.92,'precision': 0.92,'recall': 0.92}}

# Balancing data:
#{'LogisticRegression': {'accuracy': 0.44},
# 'SVC': {'accuracy': 0.24},
# 'KNeighborsClassifier': {'accuracy': 0.52},
# 'RandomForestClassifier': {'accuracy': 0.83},
# 'DecisionTreeClassifier': {'accuracy': 0.66},
# 'GradientBoostingClassifier': {'accuracy': 0.85}}

#.......................................................................................

# MFCC:

X = np.array(featuresDf['mfcc'].tolist())
# X = np.array(featuresDfBalanced['mfcc'].tolist())

# Without balancing data:
#{'LogisticRegression': {'accuracy': 0.78},
# 'SVC': {'accuracy': 0.58},
# 'KNeighborsClassifier': {'accuracy': 0.93},
# 'RandomForestClassifier': {'accuracy': 0.91},
# 'DecisionTreeClassifier': {'accuracy': 0.77},
# 'GradientBoostingClassifier': {'accuracy': 0.92}}

# Balancing data:
#{'LogisticRegression': {'accuracy': 0.71},
# 'SVC': {'accuracy': 0.48},
# 'KNeighborsClassifier': {'accuracy': 0.84},
# 'RandomForestClassifier': {'accuracy': 0.86},
# 'DecisionTreeClassifier': {'accuracy': 0.68},
# 'GradientBoostingClassifier': {'accuracy': 0.85}}

#.......................................................................................

# Fourier:

X = np.array(featuresDf['fourier'].tolist())
# X = np.array(featuresDfBalanced['fourier'].tolist())

# Without balancing data:
#{'LogisticRegression': {'accuracy': 0.56},
# 'SVC': {'accuracy': 0.52},
# 'KNeighborsClassifier': {'accuracy': 0.61},
# 'RandomForestClassifier': {'accuracy': 0.66},
# 'DecisionTreeClassifier': {'accuracy': 0.55},
# 'GradientBoostingClassifier': {'accuracy': 0.67}}

#.......................................................................................

# GradientBoostingClassifier GridSearchCV:
parameters = { 
    'n_estimators': [300, 500, 700],
    'learning_rate' :[0.3, 0.5]}
gbc = GradientBoostingClassifier() 
clf = GridSearchCV(gbc, parameters, cv=5, scoring='accuracy', verbose=5) # n_jobs= -1
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.best_score_) 
print(clf.best_params_)

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


