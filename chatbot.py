import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import random
from pathlib import Path

nltk.download('punkt')

class ChatbotPredictor:
    def __init__(self, model_path='models/chatbot_model.h5'):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.model = load_model(model_path)
            self.words = pickle.load(open('models/words.pkl', 'rb'))
            self.classes = pickle.load(open('models/classes.pkl', 'rb'))
            self.intents = self._load_intents()
        except Exception as e:
            raise Exception(f"Error al cargar el modelo: {str(e)}")

    def _load_intents(self):
        """Carga todos los archivos JSON de intenciones"""
        intents = {'intents': []}
        intent_files = list(Path('intents').glob('*.json'))
        
        if not intent_files:
            raise FileNotFoundError("No se encontraron archivos JSON en la carpeta 'intents'")
        
        for intent_file in intent_files:
            with open(intent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                intents['intents'].extend(data['intents'])
        return intents

    def clean_up_sentence(self, sentence):
        """Limpia y tokeniza la oración"""
        sentence_words = nltk.word_tokenize(sentence)
        return [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    def bag_of_words(self, sentence):
        """Crea una bolsa de palabras"""
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence, error_threshold=0.25):
        """Predice la clase de la intención"""
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        
        results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [{'intent': self.classes[r[0]], 'probability': str(r[1])} for r in results]

    def get_response(self, intents_list):
        """Obtiene una respuesta aleatoria para la intención predicha"""
        if not intents_list:
            return "No entiendo lo que quieres decir. ¿Podrías reformular tu pregunta?"
            
        tag = intents_list[0]['intent']
        
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        
        return "No tengo una respuesta para eso todavía. ¿Puedes intentar con otra pregunta?"

# Instancia global del predictor
predictor = ChatbotPredictor()

def predict_class(sentence):
    return predictor.predict_class(sentence)

def get_response(intents_list):
    return predictor.get_response(intents_list)