
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

#.............................................................................................

print('Preparing the database')
featuresDf = pd.read_pickle('./dataset/featuresDF.pkl')
featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                              featuresDf.mfcc[i]]) for i in range(len(featuresDf))]
# featuresDf.drop(columns=['sound', 'fourier', 'mfcc'], inplace=True)
df_t, df_val = train_test_split(featuresDf, test_size=0.2) # random_state=42
X = np.array(df_t['fourier_mfcc'].tolist())
y = np.array(df_t['class'].tolist())
val_x = np.array(df_val['fourier_mfcc'].tolist())
val_y = np.array(df_val['class'].tolist())
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))
val_y = np_utils.to_categorical(lb.fit_transform(val_y))
# print(X.shape) 
# print(y.shape) 

#.............................................................................................

print('Building the Neural Network')

filepath='./models/nn_checkpoint_{epoch:02d}_{val_loss:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True) # mode='max'
# earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
# callbacks_list=[checkpointer, earlystopper]
callbacks_list=[checkpointer]

num_labels = y.shape[1]

model = Sequential()

model.add(Dense(512, input_shape=(X.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(512)) # 256
model.add(Activation('relu')) # sigmoid
model.add(Dropout(0.5))

# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# 'sparse_categorical_crossentropy'

print('Training the Neural Network')
model.fit(X, y, batch_size=512, epochs=500, validation_data=(val_x, val_y), callbacks=callbacks_list)
print('Done')

# val_loss: 0.6025 - val_accuracy: 0.7990

#.............................................................................................

score = model.evaluate(val_x, val_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Test loss: 0.6634018729138925
# Test accuracy: 0.8036783337593079

#.............................................................................................

# Save the to disk so we can load it back up anytime:

# Save model weights and architecture together:
model.save('./models/nn_model_0.66_0.80.h5')

# Serialize model architecture to JSON
model_json = model.to_json()
with open("./models/nn_model_0.66_0.80.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5/H5
model.save_weights("./models/nn_weights_0.66_0.80.h5")

print('Model saved')

