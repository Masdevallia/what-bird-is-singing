
from pydub import AudioSegment
from pydub.silence import split_on_silence
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import pandas as pd
from scipy.fftpack import fft


def splitOnSilence(filepath):
    '''
    Get one track and split it where the silence is 0.1 seconds or more.
    '''
    sound = AudioSegment.from_mp3(filepath).set_channels(1) # De momento los paso todos a mono.
    # sound = AudioSegment.from_mp3(filepath).set_channels(2) # Posibilidad de pasarlos todos a estéreo
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
    - Stores the windows' array in a dataframe.
    - Calculates Fourier coefficients and stores them in the dataframe.
    - Calculates Mels coefficients and stores them in the dataframe.
    - Concatenates all features.
    - Returns the final dataframe.
    '''
    windowsList = []
    files = [f for f in listdir(filespath) if isfile(join(filespath, f))]
    for i,file in enumerate(files):
        fileID = int(re.findall(r'[0-9]+.',file)[0][:-1])
        species = re.findall(r'\w+-\w+_',file)[0][:-1]
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
                        # window = window.set_frame_rate(48000) # I can change array's length with this (48000 for convention...)
                        # Get array from each window:
                        sample = window.get_array_of_samples()
                        # Check if window has a minimum of amplitude:
                        if np.max(sample) > 1500:
                            # Fourier:
                            fft_mod = np.abs(fft(sample,512))
                            fft_mod = fft_mod[0:len(fft_mod)//2]
                            # Mels:
                            # ...
                             # Array of dictionaries:
                            windowsList.append({'class':species,'id':fileID,'sound':sample,'fourier':fft_mod,'mfcc':0})
    DF = pd.DataFrame(windowsList)
    DF.to_pickle('./dataset/featuresDF.pkl', index=False)
    return DF



# El dataframe tiene que estar balanceado (más o menos el mismo número de muestras en cada clase)



# AudioSegment(…).get_array_of_samples()
# Returns the raw audio data as an array of (numeric) samples. Note: if the audio has multiple channels,
#  the samples for each channel will be serialized – for example, stereo audio would look like 
# [sample_1_L, sample_1_R, sample_2_L, sample_2_R, …].

# AudioSegment(…).set_channels()
# Creates an equivalent version of this AudioSegment with the specified number of channels 
# (1 is Mono, 2 is Stereo). Converting from mono to stereo does not cause any audible change. 
# Converting from stereo to mono may result in loss of quality (but only if the left and right chanels differ).

# AudioSegment(…).split_to_mono()
# Splits a stereo AudioSegment into two, one for each channel (Left/Right). Returns a list with the new
#  AudioSegment objects with the left channel at index 0 and the right channel at index 1.