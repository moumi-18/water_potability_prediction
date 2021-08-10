import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction ==0:
        return render_template('index.html',output_text="Water Sample is NOT POTABLE for values{}".format(int_features),
                              )  
    else:
        return render_template('index.html',output_text="Water Sample is POTABLE for values{}".format(int_features),
                              )
    

if __name__ == "__main__":
    app.debug=True
    app.run()




