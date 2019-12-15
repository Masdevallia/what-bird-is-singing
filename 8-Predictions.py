
import subprocess
from src.audioProcessing import *

# Audio processing:
subprocess.run(['sh','./src/testAudioProcessing.sh'])
featuresDf = featuresPipeline('./dataset/test/converted/lp_ng', 'test')

