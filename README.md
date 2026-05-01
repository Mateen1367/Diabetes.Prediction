# Diabetes Prediction Project

## Overview
This project uses machine learning to predict whether a patient has diabetes based on different health-related features. The main idea is to take patient information, clean and prepare the data, then train a model to classify each patient as either diabetic or not diabetic.

This is a binary classification problem:

- `0` = No Diabetes
- `1` = Diabetes

---

## Dataset
The dataset contains medical information for each patient. Each row represents one patient, and each column represents a health-related feature.

The features used in this project include:

- `Pregnancies` — number of pregnancies
- `Glucose` — glucose level
- `BloodPressure` — blood pressure measurement
- `SkinThickness` — skin thickness measurement
- `Insulin` — insulin level
- `BMI` — body mass index
- `DiabetesPedigreeFunction` — diabetes family history score
- `Age` — patient age

The target variable is:

- `Outcome` — whether the patient has diabetes or not

---

## Objective
The goal of this project is to build a simple and understandable machine learning model that can predict diabetes outcomes.

The project focuses on:

- Exploring the dataset
- Finding patterns through visualizations
- Handling missing or unusual values
- Preparing the data for machine learning
- Training a Logistic Regression model
- Evaluating how well the model performs

---

## Tools Used
This project was completed using Python and common data science libraries:

- `Python`
- `Pandas`
- `NumPy`
- `Matplotlib`
- `Seaborn`
- `Scikit-learn`

---

## Project Steps

### 1. Data Loading and Exploration
The dataset was loaded into a pandas DataFrame. I checked the first few rows, column names, data types, and summary statistics to understand the structure of the dataset.

### 2. Data Cleaning
The data was checked for missing or unusual values. In this dataset, some medical columns may contain `0` values that do not make sense, such as glucose, blood pressure, insulin, skin thickness, or BMI. These values were handled so the model could learn from cleaner data.

### 3. Data Visualization
Charts were created to better understand the data and compare diabetes outcomes across different features. These visualizations helped show relationships between variables such as glucose, BMI, age, and the outcome.

### 4. Feature Scaling
The feature columns were scaled so they would be on a similar range. This is important for Logistic Regression because features with larger number ranges can affect the model more if the data is not scaled.

### 5. Model Training
A Logistic Regression model was trained on the prepared data. Logistic Regression was chosen because it is a good starting model for binary classification and is easier to explain than more complex models.

### 6. Model Evaluation
The model was tested using data it had not seen during training. I used evaluation metrics such as accuracy, confusion matrix, and classification report to see how well the model predicted diabetes outcomes.

---

## Results
The Logistic Regression model was able to predict diabetes outcomes with reasonable performance. The analysis showed that features like glucose, BMI, and age were important indicators in the prediction process.

The model is not meant to replace medical diagnosis. Instead, it shows how machine learning can be used to find patterns in health data and make predictions based on those patterns.

---

## Future Improvements
This project could be improved by:

- Testing more models such as Random Forest, Decision Tree, or XGBoost
- Comparing model performance side by side
- Tuning model settings to improve accuracy
- Adding more feature engineering
- Using a larger dataset with more patient information

---

## Conclusion
Overall, this project shows the basic machine learning workflow for a healthcare classification problem. The notebook loads and explores the data, cleans it, creates visualizations, scales the features, trains a Logistic Regression model, and evaluates the results.
