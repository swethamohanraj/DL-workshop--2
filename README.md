# Developing-CNN-Model-for-CIFAR-10-Dataset
## AIM:
### To develop a classification model for cifar10 data set using convolution neural network.

## ALGORITHM:
### STEP 1: Import the required packageas and import dataset using the give line: from tensorflow.keras.datasets import cifar10

### STEP 2: Split the dataset into train and test data and scale their values for reducing the model complexity.

### STEP 3: Use the onhot encoder to convert the output of train data and test data into categorical form.

### STEP 4: Build a model with convolution layer, maxpool layer and flatten it. The build a fully contected layer.

### STEP 5: Complie and fit the model. Train it and check with the test data.

### STEP 6: Check the accuracy score and make amendments if required.
## PROGRAM:
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
print(y_train.shape)
print(X_train.min())
print(X_train.max())
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
print(X_train_scaled.min())
print(X_train_scaled.max())
print(y_train[0])
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
y_train_onehot.shape
X_train_scaled = X_train_scaled.reshape(-1,32,32,3) 
X_test_scaled = X_test_scaled.reshape(-1,32,32,3)
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=28,kernel_size=(3,3),activation='relu',padding='same'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(45,activation='relu'))
model.add(layers.Dense(88))
model.add(layers.Dense(120,activation='relu'))
model.add(layers.Dense(89,activation='relu'))
model.add(layers.Dense(71))
model.add(layers.Dense(63,activation='relu'))
model.add(layers.Dense(86))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=7,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(metrics['accuracy'])
print(confusion_matrix(y_test,x_test_predictions))
```
## OUTPUT:
### Model summary:
![image](https://github.com/gpavithra673/Workshop-2-Developing-CNN-Model-for-CIFAR-10-Dataset/assets/93427264/35e79522-976d-406c-9dc3-3e97a2d7b899)
### Model accuracy vs val_accuracy:
![image](https://github.com/gpavithra673/Workshop-2-Developing-CNN-Model-for-CIFAR-10-Dataset/assets/93427264/74c27ac6-5785-447f-94b6-1358f47fdf8e)
### Model loss vs val_loss:
![image](https://github.com/gpavithra673/Workshop-2-Developing-CNN-Model-for-CIFAR-10-Dataset/assets/93427264/6c11219d-cc19-46c3-9246-b0de44eba345)
### Confusion matrix:
![image](https://github.com/gpavithra673/Workshop-2-Developing-CNN-Model-for-CIFAR-10-Dataset/assets/93427264/71fdbceb-52ab-4360-8579-970bc1ac6ff8)
### Classification report:
![image](https://github.com/gpavithra673/Workshop-2-Developing-CNN-Model-for-CIFAR-10-Dataset/assets/93427264/193ffe7a-7c7e-4b1e-83ef-2b9f53627740)

## RESULT:
### Thus we have created a classification model for Cifar10 dataset using the above given code.
