
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from src.cnnTraining import dataPreparation, accuracyPlot, lossPlot


def main():

    print('Preparing the database')
    
    featuresDf = pd.read_pickle('./dataset/featuresDF_1.pkl')
    X, y, val_x, val_y = dataPreparation(featuresDf)
    input_shape = (12, 32, 1)

    #...............................................................................

    print('Building the Neural Network')

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
    print('Model trained')

    #...............................................................................

    score = model.evaluate(val_x, val_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Test loss: 0.1545419144080981
    # Test accuracy: 0.9516493678092957

    # Epoch 2488/2500:
    # loss: 0.1986 - accuracy: 0.9277 - val_loss: 0.1403 - val_accuracy: 0.9553

    #...............................................................................

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

    #...............................................................................

    # Evaluating overfitting:
    accuracyPlot(history, 4)
    lossPlot(history, 4)


if __name__=="__main__":
    main()