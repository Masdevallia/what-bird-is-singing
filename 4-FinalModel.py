
# Final model with all the dataset:

import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from src.cnnTraining import dataPreparationFinal


def main():

    print('Preparing the database')

    featuresDf = pd.read_pickle('./dataset/featuresDF_1.pkl')
    X, y = dataPreparationFinal(featuresDf, 4)
    input_shape = (12, 32, 1)

    #...............................................................................

    print('Building the Neural Network')

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
    print('Model trained')

    #...............................................................................

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


if __name__=="__main__":
    main()