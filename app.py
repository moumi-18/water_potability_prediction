import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction ==0:
        return render_template('index.html',output_text="Water Sample is NOT POTABLE".format(prediction),
                              )  
    else:
        return render_template('index.html',output_text="Water Sample is POTABLE".format(prediction),
                              )
    

if __name__ == "__main__":
    app.run(debug=True)




