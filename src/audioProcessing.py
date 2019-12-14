
from pydub import AudioSegment
from pydub.silence import split_on_silence
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from librosa.feature import mfcc


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


def fourierCoefficients(sample):
    fft_mod = np.abs(fft(sample,512)) # probar np.log
    fft_mod = fft_mod[0:len(fft_mod)//2]
    return fft_mod


def mfccCoefficients(sample):
    mels = np.mean(mfcc(y=np.array([float(e) for e in sample]), sr=len(sample), n_mfcc=128).T, axis=0)
    return mels


def featuresPipeline(filespath):
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
    # for i,file in enumerate(files[:1]):    
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
                        sample_np = np.array(sample.tolist(), dtype=np.float64)
                        # Check if window has at least a determined amplitude:
                        if np.max(sample) > 1500:
                            # Fourier:
                            fft_mod = fourierCoefficients(sample)
                            # MFCC:
                            mels = mfccCoefficients(sample)
                             # Array of dictionaries:
                            windowsList.append({'class':species,'id':fileID,'sound':sample_np,'fourier':fft_mod,'mfcc':mels})
    DF = pd.DataFrame(windowsList)
    DF = DF[['class','id','sound','fourier','mfcc']]
    # DF['fourier_mfcc'] = [np.concatenate([DF.fourier[i], DF.mfcc[i]]) for i in range(len(DF))]
    # DF['sound-fourier_mfcc'] = [np.concatenate([DF.sound[i], DF.fourier[i], DF.mfcc[i]]) for i in range(len(DF))]
    DF.to_pickle('./dataset/featuresDF.pkl')
    return DF
