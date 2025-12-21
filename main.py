import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle

df = pd.read_csv('dataset.csv', encoding='latin1')
df = df.iloc[:, 1:]
df.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
X = df['text']
y = df['sentiment'] / 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

with open('vectorizer.pkl', 'rb') as f:
    global vectorizer
    vectorizer = pickle.load(f)

X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(X_train_vec.shape)

model = load_model('model.keras')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.evaluate(X_test_vec, y_test)
