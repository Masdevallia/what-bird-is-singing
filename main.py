
import pandas as pd
from src.data import *

# Get data:

# DF = getData('spain')
DF = pd.read_csv('./dataset/birds_spain.csv')

# speciesArray = ['European Robin', 'Iberian Green Woodpecker', 'Little Egret', 'Northern Raven']
# DF_selected = selectSpecies(speciesArray, DF)
DF_selected = pd.read_csv('./dataset/birds_spain_selected.csv')


# from pydub import AudioSegment
# audio = AudioSegment.from_mp3('./dataset/recordings/stage-1/Erithacus-rubecula_389544.mp3')