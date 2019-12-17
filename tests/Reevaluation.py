
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import model_from_json, load_model

# We can now reload the trained model whenever we want by rebuilding it and loading
# in the saved weights:

featuresDf = pd.read_pickle('./dataset/featuresDF_1.pkl')
featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                              featuresDf.mfcc[i]]) for i in range(len(featuresDf))]
df_t, df_val = train_test_split(featuresDf, test_size=0.2)
X = np.array(df_t['fourier_mfcc'].tolist())
y = np.array(df_t['class'].tolist())
val_x = np.array(df_val['fourier_mfcc'].tolist())
val_y = np.array(df_val['class'].tolist())
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))
val_y = np_utils.to_categorical(lb.fit_transform(val_y))
X = X.reshape(-1, 12, 32, 1)
val_x = val_x.reshape(-1, 12, 32, 1)


# Last epoch:
# load model (architecture + weights)
loaded_model = load_model('./models/cnn_model_epoch2500_loss0.15_accuracy0.95.h5') # returns a compiled model
# summarize model.
loaded_model.summary()
print("Model loaded from disk")
# Re-evaluate the model
loss, acc = loaded_model.evaluate(val_x,  val_y, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc)) # Restored model, accuracy: 96.79%
# scores = loaded_model.evaluate(val_x,  val_y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))


# Best epoch (checkpoint):
# load model (architecture + weights)
loaded_model = load_model('./models/cnn_checkpoint_2488_0.14.hdf5') # returns a compiled model
# summarize model.
loaded_model.summary()
print("Model loaded from disk")
# Re-evaluate the model
loss, acc = loaded_model.evaluate(val_x,  val_y, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc)) # Restored model, accuracy: 96.88%
# scores = loaded_model.evaluate(val_x,  val_y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
