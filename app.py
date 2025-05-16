from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline (ensure model.pkl is in the same directory)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Form page

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    gender = request.form['gender']
    education = request.form['education']
    course = request.form['course']
    math_score = int(request.form['math'])
    reading_score = int(request.form['reading'])

    # Create DataFrame matching the columns used during training
    data = pd.DataFrame([{
        'gender': gender,
        'parental level of education': education,
        'test preparation course': course,
        'math score': math_score,
        'reading score': reading_score
    }])

    # Make prediction
    prediction = model.predict(data)[0]

    # Return result
    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
