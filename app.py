from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
from joblib import dump, load
app = Flask(_name_)


@app.route('/', methods=['GET'])
def index():
    return "Hello World"


@app.route('/train', methods=['GET'])
def train():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=1001).fit(X, y)
   
   
    dump(clf, 'model.joblib')

    response = {
        'message': 'Model trained!'
    }
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    clf = load('model.joblib')
    req = request.get_json(force=True)
    x_test = req['data']
    x_test = np.array(x_test).reshape(1, -1)
    response = {
        'class': clf.predict(x_test).tolist()
    }
    return jsonify(response)


if _name_ == '_main_':
    app.run(host="0.0.0.0"