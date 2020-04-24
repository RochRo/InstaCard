import pandas as pd
import numpy as np
import keras
import tensorflow as tf

from Dataprep import list_of_list
from Model import get_model

data = pd.read_csv("/Users/Romain/PycharmProjects/Instacard/Data/merged_sample.csv")


L = list_of_list(data)

C = 8
T = 4
X =np.zeros(61318,T,C)


l =[]
for i in range(len(l)):
    travis = tf.keras.preprocessing.sequence.pad_sequences( L[i], maxlen= C, dtype='int32', padding='pre', truncating='pre', value=0.0 )
    travis = travis[:T]
    X[i] = travis
print(X)

X_h = X[:, :-1,:]
Y_h = X[:,-1,:]

X_train, X_test, Y_train, Y_test = train_test_split(X_h,Y_h)

model = get_model()

# fit network
history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test), verbose=2, shuffle=False)

# Accuracymodel

accuracy = model.evaluate(X_test, Y_test)
print(' Loss: {:0.4f}\n  Accuracy: {:0.4f}'.format(accuracy[0],accuracy[1]))
