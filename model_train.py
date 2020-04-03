'''https://github.com/mikesmales/Udacity-ML-Capstone/blob/master/Notebooks/2%20Data%20Preprocessing%20and%20Data%20Splitting.ipynb'''
'''https://medium.com/x8-the-ai-community/audio-classification-using-cnn-coding-example-f9cbd272269e'''

import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

train=pd.read_csv("audioData.csv")


#print(train.head())
for i in range(0,8732):
    l=train['feature'][i][1:-2].split()
    arr=np.array([float(i) for i in l])
    arr=np.reshape(arr,(7,7))
    #print(arr)
    train['feature'][i]=arr

#print(train['feature'].shape)
#print(train['feature'][0])
X = np.array(train['feature'].tolist())
y = np.array(train['class_label'].tolist())


print(X.shape)
X=np.array([arr.tolist() for arr in X.flatten()]).reshape(8732,7,7,1) #to be removed


y = to_categorical(LabelEncoder().fit_transform(y))   #or can directly use oneHot encoder

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


print(x_train.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten

num_labels = y.shape[1]
print(num_labels)
filter_size = 2

#Define Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(7,7,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels, activation='softmax'))
#Compile
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
#Train and Test The Model
model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1)
model.save("sound_classifier_model")


score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Post-training accuracy: %.4f%%" % accuracy)