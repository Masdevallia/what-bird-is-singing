
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# featuresDf = pd.read_pickle('./dataset/featuresDF.pkl')
featuresDf = pd.read_pickle('./dataset/featuresDF_2.pkl')

# El dataframe tiene que estar balanceado (más o menos el mismo número de muestras en cada clase)

#.......................................................................................

# Fourier+MFCC:

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

# Scaled:
metrics = {}
for modelName, model in models.items():
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics[modelName] = {'accuracy': round(accuracy_score(y_test, y_pred),2)}

#{'LogisticRegression': {'accuracy': 0.57},
# 'SVC': {'accuracy': 0.52},
# 'KNeighborsClassifier': {'accuracy': 0.59},
# 'RandomForestClassifier': {'accuracy': 0.9},
# 'DecisionTreeClassifier': {'accuracy': 0.77},
# 'GradientBoostingClassifier': {'accuracy': 0.93}}

#.......................................................................................

# MFCC:

X = np.array(featuresDf['mfcc'].tolist())

#{'LogisticRegression': {'accuracy': 0.78},
# 'SVC': {'accuracy': 0.58},
# 'KNeighborsClassifier': {'accuracy': 0.93},
# 'RandomForestClassifier': {'accuracy': 0.91},
# 'DecisionTreeClassifier': {'accuracy': 0.77},
# 'GradientBoostingClassifier': {'accuracy': 0.92}}

#.......................................................................................

# Fourier:

X = np.array(featuresDf['fourier'].tolist())

#{'LogisticRegression': {'accuracy': 0.56},
# 'SVC': {'accuracy': 0.52},
# 'KNeighborsClassifier': {'accuracy': 0.61},
# 'RandomForestClassifier': {'accuracy': 0.66},
# 'DecisionTreeClassifier': {'accuracy': 0.55},
# 'GradientBoostingClassifier': {'accuracy': 0.67}}

#.......................................................................................

# Sound:


#.......................................................................................

# Sound+Fourier+MFCC:
