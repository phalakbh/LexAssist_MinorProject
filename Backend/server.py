from flask import Flask, request, jsonify
import json
from tensorflow.keras.models import load_model
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
import random

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# Load the trained model and other necessary files
model = load_model('model/chatbot_model.keras')
intents = json.load(open('intents1.json'))
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))

# Helper function for text processing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    p = np.array([p])
    prediction = model.predict(p)[0]
    ERROR_THRESHOLD = 0.25
    prediction_index = np.argmax(prediction)
    probability = prediction[prediction_index]
    
    if probability > ERROR_THRESHOLD:
        return classes[prediction_index]
    else:
        return None

def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    intent = predict_class(user_input)
    if intent:
        response = get_response(intent)
    else:
        response = "Sorry, I didn't understand that. Can you please rephrase?"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
