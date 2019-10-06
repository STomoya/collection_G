'''
Test for collection_G
'''

from collection_G import send_line
# send_line("aaa", "/home/imlk/デスクトップ/IMLK/vgg16_1.png")


from collection_G import send_line_message
# send_line_message("test")


from collection_G import send_email
# send_email("a", "test message from python", "blackie0110@yahoo.co.jp")


from collection_G import encode_target
# print(encode_target(["a", "b", "a", "c"])[1])


from collection_G import plot_confusion_matrix
# from sklearn.metrics import confusion_matrix
# labels = ['a','b','c','d','e']
# plot_confusion_matrix(confusion_matrix(labels, labels, labels=labels), labels=labels)


from collection_G import plot_keras_history
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
batch_size = 128
num_classes = 10
epochs = 20
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
plot_keras_history(history)
