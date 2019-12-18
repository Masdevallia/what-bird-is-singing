
import pandas as pd
from src.cnnTraining import *


def main():

    # Preparing the database 

    # Stage 1: 4 species:
    featuresDf = pd.read_pickle('./dataset/featuresDF_1.pkl')

    # Stage 2: 10 species:
    featuresDf = pd.read_pickle('./dataset/featuresDF_2.pkl')

    X, y, val_x, val_y = dataPreparation(featuresDf)
    input_shape = (12, 32, 1)

    #...............................................................................

    # Building the Neural Network

    # Stage 1: 4 species:
    num_filters = 8
    filter_size = 3
    pool_size = 2
    batch_size = 500
    epochs = 2500

    history = cnnBuildingStage1(X, y, val_x, val_y, input_shape,
                                num_filters, filter_size, pool_size,
                                batch_size, epochs)

    # Test loss: 0.1545419144080981
    # Test accuracy: 0.9516493678092957
    # Epoch 2488/2500:
    # loss: 0.1986 - accuracy: 0.9277 - val_loss: 0.1403 - val_accuracy: 0.9553

    # Stage 2: 10 species:
    num_filters = 64
    filter_size = 3
    pool_size = 2
    batch_size = 536
    epochs = 500

    history, model, n_classes = cnnBuildingStage2(X, y,
                                val_x, val_y,
                                input_shape,
                                num_filters, filter_size, pool_size,
                                batch_size, epochs)

    # Test loss: 0.18736894733311202
    # Test accuracy: 0.9449877738952637 
    # Epoch 498/500:
    # loss: 0.1648 - accuracy: 0.9415 - val_loss: 0.1641 - val_accuracy: 0.9478

    #...............................................................................

    # ROC Curve:
    multiclassROCcurve(model, val_x, val_y, n_classes)

    # Evaluating overfitting:
    accuracyPlot(history, 4)
    lossPlot(history, 4)
    accuracyPlot(history, 10)
    lossPlot(history, 10)


if __name__=="__main__":
    main()