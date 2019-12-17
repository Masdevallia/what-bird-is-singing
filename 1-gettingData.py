
import pandas as pd
from src.getData import *

# Get data from the API:
df = getData('spain')
# df = pd.read_csv('./dataset/birds_spain.csv')

# Stage 1: 4 species
speciesArray = ['European Robin', 'Iberian Green Woodpecker', 'Little Egret', 'Northern Raven']
df_selected = selectSpecies(speciesArray, df, 1)
# df_selected = pd.read_csv('./dataset/birds_spain_selected_1.csv')


# Stage 2: 10 species
speciesArray = ['European Robin', 'Common Cuckoo', 'European Greenfinch', 'Black-winged Kite',
                'European Serin', 'Iberian Green Woodpecker', 'Little Egret', 'Little Owl',
                'Northern Raven']
df_selected = selectSpecies(speciesArray, df, 2)
# df_selected = pd.read_csv('./dataset/birds_spain_selected_2.csv')
