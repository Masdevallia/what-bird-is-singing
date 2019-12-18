
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics


def dataPreparation(featuresDf):
    print('Preparing the database') 
    featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                                 featuresDf.mfcc[i]]) for i in range(len(featuresDf))]
    df_t, df_val = train_test_split(featuresDf, test_size=0.2) # random_state=42
    X = np.array(df_t['fourier_mfcc'].tolist())
    y = np.array(df_t['class'].tolist())
    val_x = np.array(df_val['fourier_mfcc'].tolist())
    val_y = np.array(df_val['class'].tolist())
    # Mapping the string class values to integer values using a LabelEncoder.
    lb = LabelEncoder()
    y = np_utils.to_categorical(lb.fit_transform(y))
    val_y = np_utils.to_categorical(lb.fit_transform(val_y))
    X = X.reshape(-1, 12, 32, 1)
    val_x = val_x.reshape(-1, 12, 32, 1)
    return X, y, val_x, val_y


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


def accuracyPlot(history, numberClasses):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(f'./charts/cnn_accuracy_{numberClasses}classes.png', dpi=300, bbox_inches='tight')


def lossPlot(history, numberClasses):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(f'./charts/cnn_lossvalues_{numberClasses}classes.png', dpi=300, bbox_inches='tight')


def cnnBuildingStage1(X, y, val_x, val_y, input_shape, num_filters, filter_size, pool_size, batch_size, epochs):
    print('Building the Neural Network')
    filepath='./models/cnn_checkpoint_{epoch:02d}_{val_loss:.2f}.hdf5'
    checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True) # mode='max'
    # earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    # callbacks_list=[checkpointer, earlystopper]
    callbacks_list=[checkpointer]
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
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # Train the model:
    print('Training the Neural Network')
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y),
                    callbacks=callbacks_list)
    print('Model trained')
    score = model.evaluate(val_x, val_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save the to disk so we can load it back up anytime:
    # Save model weights and architecture together:
    model.save('./models/cnn_model_epoch2500_loss0.15_accuracy0.95.h5')
    # Serialize model architecture to JSON
    model_json = model.to_json()
    with open("./models/cnn_model_epoch2500_loss0.15_accuracy0.95.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5/H5
    model.save_weights("./models/cnn_model_epoch2500_loss0.15_accuracy0.95_weights.h5")
    print('Model saved')
    return history

