
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def dataPreparation(featuresDf):
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
    featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                                 featuresDf.mfcc[i]]) for i in range(len(featuresDf))]
    X = np.array(featuresDf['fourier_mfcc'].tolist())
    y = np.array(featuresDf['class'].tolist())
    # Mapping the string class values to integer values using a LabelEncoder.
    lb = LabelEncoder()
    y = np_utils.to_categorical(lb.fit_transform(y))
    np.save(f'../models/classes{numberClasses}.npy', lb.classes_)
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
    plt.savefig(f'../charts/cnn_accuracy_{numberClasses}classes.png', dpi=300, bbox_inches='tight')


def lossPlot(history, numberClasses):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(f'../charts/cnn_lossvalues_{numberClasses}classes.png', dpi=300, bbox_inches='tight')
