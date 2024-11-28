import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.load(open('intents1.json'))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize words and categorize into patterns and tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort words and classes
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes to pickle files
os.makedirs('model', exist_ok=True)  # Create 'model' directory if it doesn't exist
pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Create bag of words
    bag = [1 if word in word_patterns else 0 for word in words]

    # Output row for the current pattern
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data and convert to numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)  # Specify dtype=object to allow for subarrays

# Split training data into features (X) and labels (y)
train_x = np.array(list(training[:, 0]))  # Convert list of features to numpy array
train_y = np.array(list(training[:, 1]))  # Convert list of labels to numpy array

# Create the model using Input layer
model = Sequential()
model.add(Input(shape=(len(train_x[0]),)))  # Define input shape using Input layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model using updated parameter name
sgd = SGD(learning_rate=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)

# Save the model
model.save('model/chatbot_model.keras')
