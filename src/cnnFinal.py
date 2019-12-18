
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from sklearn import metrics


def dataPreparationFinal(featuresDf, numberClasses):
    print('Preparing the database') 
    featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                                 featuresDf.mfcc[i]]) for i in range(len(featuresDf))]
    X = np.array(featuresDf['fourier_mfcc'].tolist())
    y = np.array(featuresDf['class'].tolist())
    # Mapping the string class values to integer values using a LabelEncoder.
    lb = LabelEncoder()
    y = np_utils.to_categorical(lb.fit_transform(y))
    np.save(f'./models/classes{numberClasses}.npy', lb.classes_)
    X = X.reshape(-1, 12, 32, 1)
    return X, y


def finalCnnStage1(X, y, input_shape, num_filters, filter_size, pool_size, batch_size, epochs):
    print('Building the Neural Network')
    num_labels = y.shape[1]
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
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs)
    print('Model trained') 
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
    return history

