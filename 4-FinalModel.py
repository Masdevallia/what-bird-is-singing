
# Final model with all the dataset:

import pandas as pd
from src.cnnFinal import *


def main():

    # Preparing the database 
    
    # Stage 1: 4 species:
    featuresDf = pd.read_pickle('./dataset/featuresDF_1.pkl')
    X, y = dataPreparationFinal(featuresDf, 4)

    # Stage 2: 10 species:
    featuresDf = pd.read_pickle('./dataset/featuresDF_2.pkl')
    X, y = dataPreparationFinal(featuresDf, 10)

    input_shape = (12, 32, 1)

    #...............................................................................

    # Building the Neural Network

    # Stage 1: 4 species:
    num_filters = 8
    filter_size = 3
    pool_size = 2
    batch_size = 500
    epochs = 2500
    history = finalCnnStage1(X, y, input_shape, 
                            num_filters, filter_size, pool_size, batch_size, epochs)

    # Stage 2: 10 species:
    num_filters = 64
    filter_size = 3
    pool_size = 2
    batch_size = 536
    epochs = 500
    history = finalCnnStage2(X, y, input_shape,
                            num_filters, filter_size, pool_size, batch_size, epochs)
    

if __name__=="__main__":
    main()