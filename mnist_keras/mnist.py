import keras.utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from data_util import DataUtils
import numpy as np
import pandas as pd

model = Sequential()
model.add(Dense(500, input_shape=(28*28,)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(500))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # Set learning rate
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              class_mode='categorical', metrics=['accuracy'])

trainfile_X = './train-images-idx3-ubyte'
trainfile_y = './train-labels-idx1-ubyte'
testfile_X = './t10k-images-idx3-ubyte'
testfile_y = './t10k-labels-idx1-ubyte'
train_X = DataUtils(filename=trainfile_X).getImage()
train_y = DataUtils(filename=trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

# transform to one hot
train_y = keras.utils.to_categorical(train_y, num_classes=10)
test_y = (np.arange(10) == test_y[:, None]).astype(int)

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

predict_X = DataUtils(filename='./test-image').getImage()
print(predict_X.shape)

model.fit(train_X, train_y, batch_size=200, epochs=100, shuffle=True, verbose=1, validation_split=0.3)
print('Test set:\n')
score = model.evaluate(test_X, test_y, batch_size=200, verbose=1)
print('score: ', score, '\n')
predict_y = model.predict(predict_X, verbose=1)
predict_y = np.argmax(predict_y, axis=1)

id = np.arange(1, 10001)
predict = np.vstack((id, predict_y))
predict_df = pd.DataFrame(predict.transpose(), columns=['id', 'label'])
predict_df.to_csv('predict.csv', index=False)

