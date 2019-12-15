
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
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
# featuresDf.drop(columns=['sound', 'fourier', 'mfcc'], inplace=True)
df_t, df_val = train_test_split(featuresDf, test_size=0.2) # random_state=42
X = np.array(df_t['fourier_mfcc'].tolist())
y = np.array(df_t['class'].tolist())
val_x = np.array(df_val['fourier_mfcc'].tolist())
val_y = np.array(df_val['class'].tolist())
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))
val_y = np_utils.to_categorical(lb.fit_transform(val_y))
# print(X.shape) # (8850, 384)
# print(y.shape) # (8850, 4)

X = X.reshape(-1, 12, 32, 1)
val_x = val_x.reshape(-1, 12, 32, 1)
# print(X.shape) # (8850, 12, 32, 1)
# print(val_x.shape) # (2213, 12, 32, 1)

input_shape = (12, 32, 1)

#.............................................................................................

print('Building the Neural Network')

# Build the model:
filepath='./models/cnn_checkpoint_{epoch:02d}_{val_loss:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True) # mode='max'
# earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
# callbacks_list=[checkpointer, earlystopper]
callbacks_list=[checkpointer]

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
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model:
print('Training the Neural Network')
history = model.fit(X, y, batch_size=500, epochs=2500, validation_data=(val_x, val_y),
                   callbacks=callbacks_list)
print('Done')

#.............................................................................................

score = model.evaluate(val_x, val_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Test loss: 0.1545419144080981
# Test accuracy: 0.9516493678092957

# Epoch 2488/2500
# loss: 0.1986 - accuracy: 0.9277 - val_loss: 0.1403 - val_accuracy: 0.9553

#.............................................................................................

# Save the to disk so we can load it back up anytime:

model.save('./models/cnn_model_epoch2500_loss0.15_accuracy0.95.h5')

# serialize the model to JSON
model_json = model.to_json()
with open("./models/cnn_model_epoch2500_loss0.15_accuracy0.95.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("./models/cnn_model_epoch2500_loss0.15_accuracy0.95_weights.h5")

print('Model saved')

#.............................................................................................

import matplotlib.pyplot as plt
# %matplotlib inline

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig('./charts/cnn_accuracy.png', dpi=300, bbox_inches='tight')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig('./charts/cnn_lossvalues.png', dpi=300, bbox_inches='tight')

#.............................................................................................

# We can now reload the trained model whenever we want by rebuilding it and loading in the saved weights:

