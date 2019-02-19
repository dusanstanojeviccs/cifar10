from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = RMSprop()
NB_EPOCH = 40
BATCH_SIZE = 128

model = Sequential()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

datagen = ImageDataGenerator(
rotation_range = 40,
width_shift_range=0.2,
height_shift_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

datagen.fit(x_train)

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))


model.summary()

model.compile(loss='categorical_crossentropy', optimizer = OPTIMIZER, metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), samples_per_epoch = x_train.shape[0], epochs=NB_EPOCH, verbose=VERBOSE)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test Score: ", score[0])
print("Test Accuracy: ", score[1])