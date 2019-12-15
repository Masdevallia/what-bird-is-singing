
# Final model with all the dataset:

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#.............................................................................................

print('Preparing the database')
# featuresDf = pd.read_pickle('./dataset/featuresDF.pkl')
featuresDf = pd.read_pickle('./dataset/featuresDF_1_LN.pkl')
featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                              featuresDf.mfcc[i]]) for i in range(len(featuresDf))]

X = np.array(featuresDf['fourier_mfcc'].tolist())
y = np.array(featuresDf['class'].tolist())

# Mapping the string class values to integer values using a LabelEncoder.
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))
np.save('./models/classes.npy', lb.classes_)

print(X.shape) # (11063, 384)
print(y.shape) # (11063, 4)

X = X.reshape(-1, 12, 32, 1)
print(X.shape) # (11063, 12, 32, 1)
input_shape = (12, 32, 1)

#.............................................................................................

print('Building the Neural Network')

# Build the model:

num_labels = y.shape[1]
num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential()
model.add(Conv2D(num_filters, filter_size, input_shape=input_shape,
                 strides=2, padding='same', activation='relu'))
model.add(Conv2D(num_filters, filter_size))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile the model:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model:
print('Training the Neural Network')
history = model.fit(X, y, batch_size=500, epochs=2500)
print('Done')

#.............................................................................................

# Save the to disk so we can load it back up anytime:

# Save model weights and architecture together:
model.save('./models/cnn_model_final.h5')

# Serialize model architecture to JSON
model_json = model.to_json()
with open("./models/cnn_model_final.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5/H5
model.save_weights("./models/cnn_model_final_weights.h5")

print('Model saved')

