
import subprocess
from src.audioProcessing import *

# Audio processing:
subprocess.run(['sh','./src/audioProcessing.sh'])
featuresDf = featuresPipeline('./dataset/recordings/stage-1/converted2/lp_ng', 1)
