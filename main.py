import pandas as pd
from nltk import word_tokenize
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

tokenized_df = pd.DataFrame(columns=['text'])

for row in X:

    tokens = word_tokenize(row)
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if re.sub(r'[^a-zA-Z]', '', token)]
    tokenized_df.loc[len(tokenized_df)] = ''.join(tokens)
    
print(tokenized_df.head())