from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
sal_data = pd.read_csv('Dataset09-Employee-salary-prediction.csv')
sal_data.columns = ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_years', 'salary']
sal_data.drop_duplicates(inplace=True)
sal_data.dropna(inplace=True)

# Encode categorical data
le_gender = LabelEncoder()
le_degree = LabelEncoder()
le_job = LabelEncoder()

sal_data['Gender_Encode'] = le_gender.fit_transform(sal_data['Gender'])
sal_data['Degree_Encode'] = le_degree.fit_transform(sal_data['Degree'])
sal_data['Job_Title_Encode'] = le_job.fit_transform(sal_data['Job_Title'])

# Feature Scaling
scaler = StandardScaler()
sal_data['Age_scaled'] = scaler.fit_transform(sal_data[['Age']])
sal_data['Experience_years_scaled'] = scaler.fit_transform(sal_data[['Experience_years']])

# Prepare features and target variable
X = sal_data[['Age_scaled', 'Gender_Encode', 'Degree_Encode', 'Job_Title_Encode', 'Experience_years']]
y = sal_data['salary']

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        age = float(request.form['age'])
        gender = request.form['gender']
        degree = request.form['degree']
        job_title = request.form['job_title']
        experience_years = float(request.form['experience_years'])

        # Encode and scale inputs
        gender_encoded = le_gender.transform([gender])[0]
        degree_encoded = le_degree.transform([degree])[0]
        job_encoded = le_job.transform([job_title])[0]
        age_scaled = scaler.transform([[age]])[0][0]
        experience_scaled = scaler.transform([[experience_years]])[0][0]

        # Predict salary
        predicted_salary = model.predict([[age_scaled, gender_encoded, degree_encoded, job_encoded, experience_scaled]])[0]

        return f"<h1>Predicted Salary: ₹{round(predicted_salary, 2)}</h1>"
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
