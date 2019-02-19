from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np
import scipy.misc

model = model_from_json(open('cifar10_architecture.json').read())
model.load_weights('cifar10_weights.h5')

img_names = ['cat.jpg', 'dog.jpg']

imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)), (1, 0, 2)).astype('float32') for img_name in img_names]
imgs = np.array(imgs) / 255

optim = SGD()

model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

predictions = model.predict_classes(imgs)
print(predictions)