
from src.audioProcessing import *
import subprocess
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import sys
from keras.models import load_model
import webbrowser


def main():

    # I will receive a filename from the application:
    filename = sys.argv[1]

    # Audio processing:
    subprocess.run(['sh','./src/testAudioProcessing.sh', filename])   
    testFeaturesDf = testFeaturesPipeline('./application/uploaded/converted', filename)

    # Preparing data:
    testFeaturesDf['fourier_mfcc'] = [np.concatenate([testFeaturesDf.fourier[i],
                                testFeaturesDf.mfcc[i]]) for i in range(len(testFeaturesDf))]
    X = np.array(testFeaturesDf['fourier_mfcc'].tolist())
    X = X.reshape(-1, 12, 32, 1)

    #...............................................................................

    # Load trained model:
    loaded_model = load_model('./models/cnn_model_final.h5')

    # Class Predictions:
    ynew = loaded_model.predict_classes(X)

    # Using the LabelEncoder to convert the integers back into string values via
    # the inverse_transform() function.
    encoder = LabelEncoder()
    encoder.classes_ = np.load('./models/classes4.npy')

    # Counting how many times each species appears in the predictions:
    windowPredictions = Counter(ynew)
    # Returning the species that appears more times:
    max_key = max(windowPredictions, key=lambda x: windowPredictions[x])
    finalPrediction = encoder.inverse_transform([max_key])[0]

    #...............................................................................

    # Giving the answer:
    answer = f'./application/output/{finalPrediction}.html'
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    webbrowser.get(chrome_path).open(answer, 0)


if __name__=="__main__":
    main()