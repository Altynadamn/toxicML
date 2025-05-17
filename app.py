from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
model = load_model('models/best_model_lstm.h5')
tokenizer = joblib.load('models/tokenizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plot_url = None

    if request.method == 'POST':
        user_text = request.form['user_text']

        seq = tokenizer.texts_to_sequences([user_text])
        padded = pad_sequences(seq, maxlen=50)

        preds = model.predict(padded)[0]
        binary_preds = (preds > 0.5).astype(int)
        prediction = dict(zip(LABELS, binary_preds))

        # график
        plt.figure(figsize=(8, 4))
        plt.bar(LABELS, preds, color='salmon')
        plt.ylim(0, 1)
        plt.title("Toxicity Probabilities")
        plt.ylabel("Probability")
        plt.xticks(rotation=30)
        plt.tight_layout()

        path = os.path.join('static', 'proba_plot.png')
        plt.savefig(path)
        plt.close()
        plot_url = path

    return render_template('index.html', prediction=prediction, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
