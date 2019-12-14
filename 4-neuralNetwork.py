
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

featuresDf = pd.read_pickle('./dataset/featuresDF.pkl')

featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                              featuresDf.mfcc[i]]) for i in range(len(featuresDf))]
                     
X = np.array(featuresDf['fourier_mfcc'].tolist())
y = np.array(featuresDf['class'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2)

