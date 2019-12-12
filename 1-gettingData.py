
import pandas as pd
from src.getData import *

# Get data:

DF = getData('spain')
speciesArray = ['European Robin', 'Iberian Green Woodpecker', 'Little Egret', 'Northern Raven']
DF_selected = selectSpecies(speciesArray, DF)

# DF = pd.read_csv('./dataset/birds_spain.csv')
# DF_selected = pd.read_csv('./dataset/birds_spain_selected.csv')

