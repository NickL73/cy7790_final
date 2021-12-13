"""
Generates adversarial examples on a convolutional neural network on the CIFAR-10 dataset.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np

from art.utils import load_dataset
from art.attacks.evasion import DeepFool, CarliniL2Method, FastGradientMethod
from art.estimators.classification import KerasClassifier

tf.compat.v1.disable_eager_execution()

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:250], y_test[:250]
im_shape = x_train[0].shape

# Load into linear activation for logit use
lin = Sequential()
lin.add(Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=x_train.shape[1:]))
lin.add(Conv2D(32, (3, 3), activation='relu'))
lin.add(MaxPooling2D(pool_size=(2, 2)))
lin.add(Dropout(0.25))

lin.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
lin.add(Conv2D(64, (3, 3), activation='relu'))
lin.add(MaxPooling2D(pool_size=(2, 2)))
lin.add(Dropout(0.25))

lin.add(Flatten())
lin.add(Dense(512, activation='relu'))
lin.add(Dropout(0.5))
lin.add(Dense(10, activation='linear'))

lin.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
lin.load_weights('models/softmax_weights.h5')

# Create classifier wrapper
classifier = KerasClassifier(model=lin, use_logits=True, clip_values=(min_,max_))

# Generate DeepFool examples
adv_crafter = DeepFool(classifier)
x_test_deepfool = adv_crafter.generate(x_test)

preds = np.argmax(classifier.predict(x_test_deepfool), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy on DeepFool adversarial samples: ", (acc))

# Generate CW-L2 Examples
adv_crafter = CarliniL2Method(classifier=classifier, confidence=0.01, targeted=False)
x_test_cw = adv_crafter.generate(x_test)

preds = np.argmax(classifier.predict(x_test_cw), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy on CW adversarial samples: ", (acc))

# Generate FGSM Examples
adv_crafter = FastGradientMethod(estimator=classifier, norm=np.inf)
x_test_fgsm = adv_crafter.generate(x_test)

preds = np.argmax(classifier.predict(x_test_fgsm), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy on FGSM adversarial samples: ", (acc))

#Save all samples for future use
np.save('data/deepfool_cnn.npy', x_test_deepfool)
np.save('data/cw_cnn.npy', x_test_cw)
np.save('data/fgsm_cnn.npy', x_test_fgsm)

########## REPEAT FOR RESNET
# Create classifier wrapper
res = keras.models.load_model('models/resnet.h5')
classifier = KerasClassifier(model=res, use_logits=True, clip_values=(min_,max_))

# Generate DeepFool examples
adv_crafter = DeepFool(classifier)
x_test_deepfool = adv_crafter.generate(x_test)

preds = np.argmax(classifier.predict(x_test_deepfool), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy on DeepFool adversarial samples: ", (acc))

# Generate CW-L2 Examples
adv_crafter = CarliniL2Method(classifier=classifier, confidence=0.01, targeted=False)
x_test_cw = adv_crafter.generate(x_test)

preds = np.argmax(classifier.predict(x_test_cw), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy on CW adversarial samples: ", (acc))

# Generate FGSM Examples
adv_crafter = FastGradientMethod(estimator=classifier, norm=np.inf)
x_test_fgsm = adv_crafter.generate(x_test)

preds = np.argmax(classifier.predict(x_test_fgsm), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy on FGSM adversarial samples: ", (acc))

#Save all samples for future use
np.save('data/deepfool_res.npy', x_test_deepfool)
np.save('data/cw_res.npy', x_test_cw)
np.save('data/fgsm_res.npy', x_test_fgsm)
