from tensorflow.keras.models import load_model
import pickle
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from numpy import argmax
import starlette.status as status

app = FastAPI()

with open('vectorizer.pkl', 'rb') as f:
    global vectorizer
    vectorizer = pickle.load(f)


model = load_model('model.keras')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

@app.get('/')
async def root():
    return RedirectResponse(url='/predict', status_code=status.HTTP_200_OK)

class Text(BaseModel):
    text: str

@app.post('/predict')
async def predict(prediction: Text):
    text_to_predict = prediction.text
    text_vec = vectorizer.transform([text_to_predict])
    verdict = model.predict(text_vec)[0]
    labels = ['negative', '','positive']
    return {
        "prediction": labels[argmax(verdict)],
        "values": {
            "positive": float(verdict[-1]),
            "negative": float(verdict[0])
        }
    }