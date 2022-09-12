import json
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():   
    if model:
        int_features = [x for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)[0]
        
        return render_template('index.html', prediction_text = prediction)
        

if __name__ == "__main__":
    app.run(host="127.0.0.2",port=3000, debug=True)