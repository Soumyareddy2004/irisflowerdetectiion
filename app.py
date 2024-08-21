from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
model = pickle.load(open('iris.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    
    # Prepare the data for prediction
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    
    # Render the prediction result page
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)
