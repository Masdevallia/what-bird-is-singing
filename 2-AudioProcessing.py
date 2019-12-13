
from src.audioProcessing import *

# Applying low pass filter + noise gate:
import subprocess
subprocess.call(['./src/audioProcessing.sh'])

# Separating out silent chunks and split each remaining chunk into 1 second windows overlapping by 50%.

windowsDF('./dataset/recordings/stage-1/converted')







# from pydub import AudioSegment
# audio = AudioSegment.from_mp3('./dataset/recordings/stage-1/Erithacus-rubecula_389544.mp3')

# for each window, determine the average value of each mfcc coefficient. (so you have 16 features for each window)
# use a random forest classifier
# average the probabilities for each window to get the probability for the entire clip


