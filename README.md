# Diabetes Prediction Project

**One-Sentence Description:**  
This project uses tabular patient health data and machine learning to predict whether a person is likely to have diabetes.

---

## Project Overview
This project is a tabular classification machine learning project based on a diabetes dataset. The goal is to predict whether a patient has diabetes using health-related features such as glucose, BMI, age, insulin, and blood pressure.

This is a binary classification problem:

- `0` = No Diabetes
- `1` = Diabetes

The project follows the main steps of a data science workflow: defining the problem, loading the data, exploring the dataset, cleaning and preparing the data, creating visualizations, training a machine learning model, evaluating performance, and preparing results.

---

## Project Link
Kaggle Dataset / Challenge Link:  
_https://www.kaggle.com/competitions/diabetes-prediction-with-nn/rules
_

---

## Dataset Description
The dataset contains patient health information in tabular format. Each row represents one patient, and each column represents a health-related feature.

The features used in this project include:

- `Pregnancies` — number of pregnancies
- `Glucose` — glucose level
- `BloodPressure` — blood pressure measurement
- `SkinThickness` — skin thickness measurement
- `Insulin` — insulin level
- `BMI` — body mass index
- `DiabetesPedigreeFunction` — diabetes family history score
- `Age` — patient age
- `Outcome` — target variable showing diabetes status

The target variable is `Outcome`.

---

## Problem Type
This project is a classification problem because the model predicts a category instead of a continuous number.

The target is encoded as:

- `0` = patient does not have diabetes
- `1` = patient has diabetes

---

## Data Loading and Initial Exploration
The dataset was loaded into a pandas DataFrame. After loading the data, I checked:

- The number of rows and columns
- The column names
- The data types
- Summary statistics
- Missing or unusual values
- Class balance between diabetic and non-diabetic patients

This helped me understand the structure of the dataset before cleaning and modeling.

---

## Feature Summary
The dataset contains mostly numerical features. These features represent medical measurements or patient information.

| Feature | Type | Description |
|---|---|---|
| Pregnancies | Numerical | Number of pregnancies |
| Glucose | Numerical | Glucose level |
| BloodPressure | Numerical | Blood pressure measurement |
| SkinThickness | Numerical | Skin thickness measurement |
| Insulin | Numerical | Insulin level |
| BMI | Numerical | Body mass index |
| DiabetesPedigreeFunction | Numerical | Diabetes family history score |
| Age | Numerical | Patient age |
| Outcome | Categorical | Target variable: 0 or 1 |

---

## Missing Values and Outliers
The dataset was checked for missing values and unusual values. Some columns had `0` values that may not make sense medically, such as:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

These values were treated as unusual or missing values and handled during data cleaning.

Outliers were also checked by looking at feature distributions and summary statistics. In this project, an outlier means a value that is much higher or lower than most of the other values in the same feature.

---

## Class Balance
Since this is a classification problem, I checked how many patients were in each class:

- Class `0`: No Diabetes
- Class `1`: Diabetes

This step is important because if one class is much larger than the other, the model may become biased toward predicting the majority class.

---

## Data Visualization
Visualizations were created to better understand the dataset and compare features between diabetes outcomes.

The graphs helped show patterns in features such as:

- Glucose
- BMI
- Age
- Insulin
- Blood Pressure

These visualizations helped identify which features seemed most useful for predicting diabetes. Features like glucose, BMI, and age appeared to show stronger differences between diabetic and non-diabetic patients.

---

## Data Cleaning and Preparation
Before training the model, the data was prepared for machine learning.

The preparation steps included:

- Checking for missing or unusual values
- Handling unrealistic `0` values in medical columns
- Separating the input features from the target column
- Making sure the target variable was correctly encoded
- Scaling the numerical features
- Splitting the data into training, validation, and test sets

Feature scaling was used because Logistic Regression can be affected by features having very different number ranges.

---

## Machine Learning Model
The machine learning model used in this project was:

- Logistic Regression

Logistic Regression was chosen because it is a good starting model for binary classification problems. It is also easier to explain compared to more complex models.

The model was trained using the cleaned and scaled training data.

---

## Model Evaluation
The model was evaluated using data it had not seen during training.

The evaluation included:

- Accuracy
- Confusion matrix
- Classification report
- Precision
- Recall
- F1-score

These metrics helped show how well the model predicted both diabetes and non-diabetes outcomes.

---

## Results
The Logistic Regression model was able to make reasonable predictions for diabetes outcomes. The model showed that health features can be used to find patterns related to diabetes.

Some of the most useful indicators were:

- Glucose
- BMI
- Age

The results show that machine learning can help identify patterns in healthcare data, but the model should not be treated as a medical diagnosis. It is only a prediction tool based on the dataset.

---

## Future Improvements
This project could be improved by:

- Testing more models such as Decision Tree, Random Forest, or XGBoost
- Comparing model performance side by side
- Tuning hyperparameters
- Improving feature engineering
- Trying different ways to handle missing or unusual values
- Using a larger dataset with more patient information

---

## Conclusion
- Overall, this project demonstrates the main steps of a tabular machine learning classification project. The notebook loads the data, explores the features, checks for missing values and outliers, creates visualizations, prepares the data, trains a Logistic Regression model, and evaluates the results.
---

