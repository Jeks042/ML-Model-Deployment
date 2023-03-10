#!/usr/bin/env python3.9

import pickle
from flask import Flask, render_template, request, jsonify

# Load the model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define a route to handle the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Print the input data received from the form
    print(f"Input Data: [{sepal_length}, {sepal_width}, {petal_length}, {petal_width}]")

    # Make a prediction using the model
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    species = prediction[0]

    # Return the result to the user
    return render_template('result.html', sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width, species=species)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
