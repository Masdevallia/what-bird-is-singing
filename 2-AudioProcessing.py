
from os import listdir
from os.path import isfile, join
import re
from src.audioProcessing import splitOnSilence


# Applying low pass filter + noise gate:
import subprocess
subprocess.call(['./src/audioProcessing.sh'])

# split_on_silence function for separating out silent chunks:
mypath = './dataset/recordings/stage-1/converted'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i,file in enumerate(files):
    fileID = int(re.findall('[0-9]+.',file)[0][:-1])
    chunks = splitOnSilence(f'{mypath}/{file}')
    print(f'Splitting {file}: file {i+1} out of {len(files)}')
    for chunk in chunks:
        # Define windows with overlap
        # ...
        # For each window:      
        sample = window.get_array_of_samples()

# split each clip into 1 second windows overlapping by 50%.









# from pydub import AudioSegment
# audio = AudioSegment.from_mp3('./dataset/recordings/stage-1/Erithacus-rubecula_389544.mp3')

# for each window, determine the average value of each mfcc coefficient. (so you have 16 features for each window)
# use a random forest classifier
# average the probabilities for each window to get the probability for the entire clip


