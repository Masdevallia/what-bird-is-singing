

# Applying low pass filter + noise gate:
import subprocess
subprocess.call(['./src/audioProcessing.sh'])


# Separating out silent chunks.
# Split each remaining chunk into 1 second windows overlapping by 50%.
# Calculating Fourier coefficients for each window.
# Determining the average value of each mfcc coefficient for each window.
from src.audioProcessing import *
featuresDf = featuresPipeline('./dataset/recordings/stage-2/converted', 2)
# featuresDf = pd.read_pickle('./dataset/featuresDF.pkl')
