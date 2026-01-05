from flask import Flask, request, render_template
import os
import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import text cleaner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import clean_text

# Initialize app
app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = os.path.join("..", "models", "lstm_model.keras")
TOKENIZER_PATH = os.path.join("..", "models", "tokenizer.pkl")

model = load_model(MODEL_PATH, compile=False)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    score = None

    if request.method == "POST":
        user_input = request.form.get("user_input")
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100, padding="post")
        pred = model.predict(padded)[0][0]
        score = round(float(pred), 4)
        prediction = "Positive " if pred >= 0.5 else "Negative "

    return render_template("index.html", prediction=prediction, score=score)

if __name__ == "__main__":
    app.run(debug=True)
