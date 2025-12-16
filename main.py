import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
# import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import string
import re

df = pd.read_csv('dataset.csv')

# nltk.download('all')     # If 'punkt_tab' is missing, run: nltk.download('punkt_tab')

print("\n", df.head())

X = df['text']
y = df['sentiment']

print(f"\nX:\n{X.head()}\ny:\n{y.head()}\n")

example = X.iloc[1]

tokens = word_tokenize(example)
stop_words = set(stopwords.words("english"))

tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if re.sub(r'[^a-zA-Z]', '', token)]

processed_example = ''.join(tokens)
print(processed_example)