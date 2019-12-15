
import subprocess
from src.audioProcessing import *
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Audio processing:
subprocess.run(['sh','./src/testAudioProcessing.sh'])
featuresDf = featuresPipeline('./dataset/test/converted/lp_ng', 'test')
# featuresDf = pd.read_pickle('./dataset/featuresDF_test_LN.pkl')

# Tengo que hacer otra función para el test que recoja sólo X.
# Tiene que guardar también una ID (nombre del fichero) para luego poder hacer la media
# de las ventanas del audio.

# Preparing data:
featuresDf['fourier_mfcc'] = [np.concatenate([featuresDf.fourier[i],
                              featuresDf.mfcc[i]]) for i in range(len(featuresDf))]
X = np.array(featuresDf['fourier_mfcc'].tolist())
X = X.reshape(-1, 12, 32, 1)

# Load trained model:
loaded_model = load_model('./models/cnn_model_final.h5')

# Class Predictions:
ynew = loaded_model.predict_classes(X)

# Using the LabelEncoder to convert the integers back into string values via
# the inverse_transform() function.
encoder = LabelEncoder()
encoder.classes_ = np.load('./models/classes.npy')
encoder.inverse_transform([0])

encoder.classes_
# ['Corvus-corax', 'Egretta-garzetta', 'Erithacus-rubecula','Picus-sharpei']
# 0: Corvus-corax
# 1: Egretta-garzetta
# 2: Erithacus-rubecula
# 3: Picus-sharpei

# Probability Predictions:
# Predicts the probability of the data instance  belonging to each class.
ynew = loaded_model.predict_proba(X) # ynew=loaded_model.predict(X)

# The output of our network is 4 probabilities (because of softmax),
# so we can use np.argmax() to turn those into actual classes.
print(np.argmax(ynew, axis=1))

