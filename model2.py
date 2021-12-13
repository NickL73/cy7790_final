import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

import numpy as np
from tensorflow.keras.applications import ResNet50


batch_size = 128
nb_classes = 10
nb_epoch = 5

# input image dimensions
img_rows, img_cols = 32, 32

# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

def preprocess_image_input(input_images):
    input_images = input_images.astype('float32')
    output_ims = tensorflow.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims

train_X = preprocess_image_input(X_train)
valid_X = preprocess_image_input(X_test)

def feature_extractor(inputs):
    ext = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))(inputs)
    return ext

def classifier(inputs):
	x = layers.GlobalAveragePooling2D()(inputs)
	x = layers.Flatten()(x)
	x = layers.Dense(1024, activation='relu')(x)
	x = layers.Dense(512, activation='relu')(x)
	x = layers.Dense(10, activation='softmax', name="classification")(x)

	return x

def model(inputs):
	resize = layers.UpSampling2D(size=(7,7))(inputs)
	res = feature_extractor(resize)
	classify = classifier(res)

	return classify

inp = layers.Input(shape=(32,32,3))
out = model(inp)
model = tensorflow.keras.Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True)

predictions = model.predict(valid_X)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

model.save_weights('models/resnet_weights.h5')
