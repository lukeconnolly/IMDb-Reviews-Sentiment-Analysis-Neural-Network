# Luke Connolly
# December 16, 2018

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.model_selection import KFold

# Parameters for training and testing
num_splits = 5
num_epochs = 4

vocabulary_size = 5000 # Top vocabulary_size most common words to be looked at
max_words = 500 # Number of words to limit a single review to

# Import the reviews. Pad them with 0s if they have less than 500 words.
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words = vocabulary_size)
X_train = sequence.pad_sequences(X_train, maxlen = max_words)
X_test = sequence.pad_sequences(X_test, maxlen = max_words)

# Create the model
model = Sequential() # Feed forward network
model.add(Embedding(vocabulary_size, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(150, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Merge the train and test datasets since doing k-fold cross validation anyway
X = np.concatenate((X_train,X_test),axis=0)
Y = np.concatenate((Y_train,Y_test),axis=0)

kf = KFold(n_splits=num_splits) # K fold cross validation with numSplts splits

accuracies = [] # Empty array to hold accuraies from each split
accuracy = 0    # Variable that will help to calculate accuracy later

# Store the original weight initialization so we can reset them every split
origWeights = model.get_weights()

# Train and evaluate the network with k-fold cross validation
for train_index, test_index in kf.split(X):
    model.set_weights(origWeights) # Reset weights
    Kx_train, Kx_test = X[train_index], X[test_index] # Training data for this iteration
    Ky_train, Ky_test = Y[train_index], Y[test_index] # Testing data for this iteration
    # Train the network
    model.fit(Kx_train, Ky_train, validation_data=(Kx_test, Ky_test), epochs=num_epochs, batch_size=128, verbose=1)
    results = model.evaluate(Kx_test, Ky_test, verbose=0) # Test accuracy for this iteration
    accuracies.append(results[1]*100) # Append the test set's accuracy to this 'accuracies' array

# Add up all the accuracies from each k-fold iteration, then divide by num_splits
for i in accuracies:
    accuracy += i
accuracy /= num_splits

# Output the overall accuracy
print("Overall Accuracy: %.2f%%" % accuracy)