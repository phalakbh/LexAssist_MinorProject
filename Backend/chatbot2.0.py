import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import tensorflow as tf

# Suppress TensorFlow logging messages
tf.get_logger().setLevel('ERROR')

# Disable TensorFlow progress bars
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disables most TF logging

# Initialize the lemmatizer and load data
lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents1.json'))
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    # Set verbose to 0 to suppress progress bars
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.55

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    # If no intent has a probability above the threshold, return a default response
    if not return_list:
        return_list.append({'intent': 'no_match', 'probability': '0'})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I'm sorry, I didn't understand that."

    return result

while True:
    message = input("You: ")

    # Allow user to exit the loop
    if message.lower() in ["exit", "quit","I am done","thanks!"]:
        print("Thank you! Let me know if you have any further questions.")
        break

    ints = predict_class(message)
    res = get_response(ints, intents)

    # Print only the bot's response, without confidence level or additional logs
    print(f"Bot: {res} (Confidence: {ints[0]['probability']})")
