
from pydub import AudioSegment
from pydub.silence import split_on_silence
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import pandas as pd


def splitOnSilence(filepath):
    '''
    Get one track and split it where the silence is 0.1 seconds or more.
    '''
    sound = AudioSegment.from_mp3(filepath)
    chunks = split_on_silence(sound,
    # Specify that a silent chunk must be at least 0.1 second or 100 ms long.
    min_silence_len = 100,
    # Consider a chunk silent if it's quieter than sound's mean dB - 16 dBFS.
    silence_thresh = sound.dBFS-16,
    # Don't keep silence at the beginning and end of the chunk
    keep_silence=0)
    return chunks

def windowsDF(filespath):
    '''
    For all tracks in filespath:
    - Separates out silent chunks.
    - Splits each remaining chunk into 1 second windows overlapping by 50%.
    - Store the windows' array in a dataframe.
    '''
    windowsList = []
    files = [f for f in listdir(filespath) if isfile(join(filespath, f))]
    for i,file in enumerate(files):
        fileID = int(re.findall('[0-9]+.',file)[0][:-1])
        species = re.findall('\w+-\w+_',file)[0][:-1]
        # Get one track and split it where the silence is 0.1 seconds or more.
        chunks = splitOnSilence(f'{filespath}/{file}')
        print(f'Splitting {file}: file {i+1} out of {len(files)}')
        for chunk in chunks:
            # Define windows with overlap:
            windowLen = 1000
            overlap = 500
            if len(chunk) >= 1000:
                for i in range(0,len(chunk),overlap):
                    if i+windowLen <= len(chunk):
                        window = chunk[i:i+windowLen]
                        # window = window.set_frame_rate(16000) # I can reduce array length with this.
                        # Get array from each window:
                        sample = window.get_array_of_samples()
                        # Check if window has a minimum of amplitude/dB...
                        # ...
                        # Array of dictionaries:
                        windowsList.append({'class':species,'id':fileID,'sound':sample,'fourier':0,'mfcc':0})
