
from pydub import AudioSegment
from pydub.silence import split_on_silence

def splitOnSilence(filepath):
    '''
    Get track and split it where the silence is 0.1 seconds or more.
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
