import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
import random
from pathlib import Path

class ChatbotTrainer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', '.', ',']
        
    def load_intents(self):
        """Carga todos los archivos JSON de la carpeta intents"""
        combined_intents = {'intents': []}
        intent_files = list(Path('intents').glob('*.json'))
        
        if not intent_files:
            raise FileNotFoundError("No se encontraron archivos JSON en la carpeta 'intents'")
        
        for intent_file in intent_files:
            with open(intent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_intents['intents'].extend(data['intents'])
        return combined_intents
    
    def prepare_data(self):
        """Prepara los datos para el entrenamiento"""
        intents = self.load_intents()
        
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        self.words = [self.lemmatizer.lemmatize(w.lower()) 
                     for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        # Crear carpeta models si no existe
        Path('models').mkdir(exist_ok=True)
        
        pickle.dump(self.words, open('models/words.pkl', 'wb'))
        pickle.dump(self.classes, open('models/classes.pkl', 'wb'))
        
    def create_training_data(self):
        """Crea los datos de entrenamiento"""
        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            bag = []
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in doc[0]]
            
            for word in self.words:
                bag.append(1) if word in pattern_words else bag.append(0)
            
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        random.shuffle(training)
        train_x = np.array([row[0] for row in training])
        train_y = np.array([row[1] for row in training])
        
        return train_x, train_y
    
    def build_model(self, input_shape, output_shape):
        """Construye el modelo de red neuronal"""
        model = Sequential([
            Dense(128, input_shape=(input_shape,), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(output_shape, activation='softmax')
        ])
        
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=0.9)
        
        sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', 
                     optimizer=sgd, 
                     metrics=['accuracy'])
        return model
    
    def train(self, epochs=200, batch_size=5):
        """Entrena el modelo"""
        self.prepare_data()
        train_x, train_y = self.create_training_data()
        
        model = self.build_model(len(train_x[0]), len(train_y[0]))
        hist = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
        
        model.save('models/chatbot_model.h5')
        print("Modelo entrenado y guardado en 'models/chatbot_model.h5'")
        return model

if __name__ == "__main__":
    try:
        trainer = ChatbotTrainer()
        trainer.train()
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")