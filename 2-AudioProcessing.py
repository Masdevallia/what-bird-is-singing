
from src.audioProcessing import *

# Applying low pass filter + noise gate:
import subprocess
subprocess.call(['./src/audioProcessing.sh'])

# Separating out silent chunks and split each remaining chunk into 1 second windows overlapping by 50%.
# Fourier.
# For each window, determine the average value of each mfcc coefficient.
DF = windowsDF('./dataset/recordings/stage-1/converted')


