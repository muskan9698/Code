# Importing datasets

import pandas
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


X=[]
num_inputs = 784
num_samples = len(train)

# Converting flattened pixels of train image into 2d matrix of (28,28,1) shape

for i in np.arange(0,num_samples):
    pixels = train.iloc[i][1:]
    pixels = np.array(pixels, dtype = int)
    pixels = np.reshape(pixels, (28,28,1))
    pixels = pixels / 255
    X.append(pixels)
X = np.array(X)
y=train["label"].values

test_2d=[]
num_inputs = 784
num_samples = len(test)

#Converting flattened pixels of test image into 2d matrix of (28,28,1) shape


for i in np.arange(0,num_samples):
  pixels = test.iloc[i]
    pixels = np.array(pixels, dtype = int)
    pixels = np.reshape(pixels, (28,28,1))
    pixels = pixels / 255
    test_2d.append(pixels)
test_2d = np.array(test_2d)


from keras.models import Sequential
import tensorflow as tf


#Creating Keras Sequential model to implement CNN

model = Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10000, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

# compile the keras model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
#Splitting training data into training and validation data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.33, random_state=42)

# fit the keras model on the dataset
model.fit(X_train, y_train,batch_size=128, epochs=10, validation_data=(X_test, y_test))

accuracy = model.evaluate(X, y)

#Prediction for test data

import numpy as np
import matplotlib.pyplot as plt
import csv
p=model.predict(test_2d)

#Writing predictions into an output csv file

fields=["ImageId","Label"]
rows=[]
j=1
for i in p:
    rows.append([j,np.argmax(i)])
    j=j+1

with open("./output.csv", 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)


