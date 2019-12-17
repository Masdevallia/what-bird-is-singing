
import subprocess
from src.audioProcessing import *

# Audio processing:

# Stage 1: 4 species:
subprocess.run(['sh','./src/audioProcessing.sh','./dataset/recordings/stage-1'])
featuresDf = featuresPipeline('./dataset/recordings/stage-1/converted', 1)

# Stage 2: 10 species:
subprocess.run(['sh','./src/audioProcessing.sh','./dataset/recordings/stage-2'])
featuresDf = featuresPipeline('./dataset/recordings/stage-2/converted', 2)
