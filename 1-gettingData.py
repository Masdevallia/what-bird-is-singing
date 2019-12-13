
import pandas as pd
from src.getData import *

# Get data:

df = getData('spain')
speciesArray = ['European Robin', 'Iberian Green Woodpecker', 'Little Egret', 'Northern Raven']
df_selected = selectSpecies(speciesArray, df)

# df = pd.read_csv('./dataset/birds_spain.csv')
# df_selected = pd.read_csv('./dataset/birds_spain_selected.csv')

