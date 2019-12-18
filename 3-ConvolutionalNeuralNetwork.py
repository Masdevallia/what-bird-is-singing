
import pandas as pd
from src.cnnTraining import *


def main():

    # Preparing the database 
    # featuresDf = pd.read_pickle('./dataset/featuresDF_1.pkl')
    featuresDf = pd.read_pickle('./dataset/featuresDF_2.pkl')
    X, y, val_x, val_y = dataPreparation(featuresDf)
    input_shape = (12, 32, 1)

    #...............................................................................

    # Building the Neural Network
    num_filters = 8
    filter_size = 3
    pool_size = 2
    batch_size=500
    epochs=2500

    history = cnnBuildingStage1(X, y, val_x, val_y, input_shape,
                                num_filters, filter_size, pool_size,
                                batch_size, epochs)

    # Stage 1: 4 species:
    # Test loss: 0.1545419144080981
    # Test accuracy: 0.9516493678092957
    # Epoch 2488/2500:
    # loss: 0.1986 - accuracy: 0.9277 - val_loss: 0.1403 - val_accuracy: 0.9553

    #...............................................................................

    # Evaluating overfitting:
    accuracyPlot(history, 4)
    lossPlot(history, 4)
    accuracyPlot(history, 10)
    lossPlot(history, 10)

    #...............................................................................

    def cnnBuildingStage2(X, y, val_x, val_y):
        print('Building the Neural Network')
        filepath='./models/stage2_cnn_checkpoint_{epoch:02d}_{val_loss:.2f}.hdf5'

        model.save('./models/stage2_cnn_model_epoch2500.h5')
        with open("./models/stage2_cnn_model_epoch2500.json", "w") as json_file:
            json_file.write(model_json) 
        model.save_weights("./models/stage2_cnn_model_epoch2500_weights.h5")


    #...............................................................................

    # Stage 2: 10 species:
    # Test loss: 0.3584835588640572
    # Test accuracy: 0.8810513615608215 
    # Epoch 2474/2500:
    # loss: 0.5333 - accuracy: 0.8121 - val_loss: 0.3569 - val_accuracy: 0.8814




if __name__=="__main__":
    main()