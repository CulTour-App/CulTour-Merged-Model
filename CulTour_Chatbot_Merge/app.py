from flask import Flask, request, jsonify
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random
import pickle
from keras.models import load_model
import warnings

# Mengabaikan warning
warnings.filterwarnings('ignore')

# Mengunduh package yang dibutuhkan dari NLTK
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inisialisasi Flask app
app = Flask(__name__)

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# Memuat data model dan file pickle yang telah disimpan
model = load_model('chatbot_model.h5')
intents = json.load(open('intents.json', encoding='utf-8'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Fungsi Preprocessing Input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

# Fungsi Prediksi dan Respon Chatbot
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res

# Route untuk Chatbot API
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    response = chatbot_response(user_message)
    return jsonify({"response": response})

# Menjalankan server
if __name__ == '__main__':
    app.run(debug=True)
