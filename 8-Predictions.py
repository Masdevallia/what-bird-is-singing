

# import subprocess
# from src.audioProcessing import *
# from sklearn.preprocessing import LabelEncoder
# from collections import Counter
# import sys
# from keras.models import load_model

# I will receive a filename from the application:
filename = sys.argv[1]
print(filename)

# Audio processing:
subprocess.run(['sh','./src/testAudioProcessing.sh'])
testFeaturesDf = testFeaturesPipeline('./dataset/test/converted/lp_ng', filename)
# testFeaturesDf = pd.read_pickle('./dataset/featuresDF_test_LN.pkl')

# Preparing data:
testFeaturesDf['fourier_mfcc'] = [np.concatenate([testFeaturesDf.fourier[i],
                              testFeaturesDf.mfcc[i]]) for i in range(len(testFeaturesDf))]
X = np.array(testFeaturesDf['fourier_mfcc'].tolist())
X = X.reshape(-1, 12, 32, 1)

# Load trained model:
loaded_model = load_model('./models/cnn_model_final.h5')

# Class Predictions:
ynew = loaded_model.predict_classes(X)

# Using the LabelEncoder to convert the integers back into string values via
# the inverse_transform() function.
encoder = LabelEncoder()
encoder.classes_ = np.load('./models/classes.npy')

# encoder.classes_
# 0: Corvus-corax
# 1: Egretta-garzetta
# 2: Erithacus-rubecula
# 3: Picus-sharpei

# Counting how many times each species appears in the predictions:
windowPredictions = Counter(ynew)
# Returning the species that appears more times:
max_key = max(windowPredictions, key=lambda x: windowPredictions[x])
finalPrediction = encoder.inverse_transform([max_key])[0]

# Probability Predictions:
# Predicts the probability of the data instance  belonging to each class.
# ynew = loaded_model.predict_proba(X) # ynew=loaded_model.predict(X)
# The output of our network is 4 probabilities (because of softmax),
# so we can use np.argmax() to turn those into actual classes.
# print(np.argmax(ynew, axis=1))

print(finalPrediction)
