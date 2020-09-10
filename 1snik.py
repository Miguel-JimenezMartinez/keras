import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
#from matplotlib import pyplot as plt

#Cargando los datos

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

'''
X_train.shape
y=y_train[0:12]
print(X_train.shape)
print(y)'''

#preocesando los datos pa que quepan :v

X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')

X_train /= 255
X_valid /= 255

#x=X_valid[0]
#print(x)

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)

'''
y=y_valid[0]
print(y)
'''

#Diseñando la red

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

#model.summary()

#Configurando la red

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

#entrenando

model.fit(X_train, y_train, batch_size=128, epochs=2, verbose=1, validation_data=(X_valid, y_valid))

