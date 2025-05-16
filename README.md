# Student Admission Predictor Web App 🎓

This project is a **Flask-based machine learning web application** that predicts student admission chances (or similar predictions) based on user input features like course, college, gender, etc. The model is trained using **Scikit-learn**, and the interface is 
built with **Flask and HTML**.

## 🚀 Features

- Predicts outcomes based on user input
- Machine learning model serialized using `pickle`
- Clean and simple HTML interface
- Powered by Flask backend for request handling

---

## 🧠 Machine Learning Model

The model is trained using Scikit-learn and includes a preprocessing pipeline using:
- OneHotEncoder for categorical variables
- StandardScaler for numerical variables
- A regression or classification model (e.g., Linear Regression, RandomForest)

> Make sure to include or mention the `train_model.py` script that handles model training.

---

## 🗂️ Project Structure
flask-web-app/
│
├── app.py # Flask app main file
├── model.pkl # Trained ML model
├── train_model.py # Python script to train and export model
├── templates/
│ ├── index.html # HTML input form
│ └── result.html # Prediction result display
└── README.md # Project documentation

📊 Dataset
The dataset used for training the model is from kaggle.

Example:

The model is trained on a dataset containing student admission-related features. (e.g., gender, course, college type, etc.)

🛠️ Technologies Used
Python 3.10

Flask
Scikit-learn
Pandas
HTML, CSS (basic)

✍️ Author
Narendra Padavala
Computer Science Engineering Student
