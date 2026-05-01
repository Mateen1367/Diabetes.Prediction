# Diabetes Prediction Project

This project uses tabular patient health data and machine learning to predict whether a person is likely to have diabetes.

---

## Project Overview
This project is a tabular classification machine learning project based on a diabetes dataset. The goal is to predict whether a patient has diabetes using health-related features such as glucose, BMI, age, insulin, and blood pressure.

This is a binary classification problem:

- `0` = No Diabetes
- `1` = Diabetes

The project follows the main steps of a data science workflow: defining the problem, loading the data, exploring the dataset, cleaning and preparing the data, creating visualizations, training machine learning models, evaluating performance, and reviewing important features.

---

## Project Link
Kaggle Dataset / Challenge Link:  
https://www.kaggle.com/competitions/diabetes-prediction-with-nn/rules

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

Example code used to load and inspect the dataset:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("diabetes.csv")

df.shape
df.info()
df.describe()
df.isnull().sum()
```

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

A feature summary table was also created in the notebook to show each column’s data type, missing values, and number of unique values.

```python
summary = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

summary
```

---

## Missing Values and Outliers
The dataset was checked for missing values and unusual values. At first, the dataset did not show regular blank missing values. However, some medical columns had `0` values that may not make sense medically, such as:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

These values were treated as unusual or missing values and handled during data cleaning.

```python
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.median(), inplace=True)
```

Outliers were also checked by looking at feature distributions and summary statistics. In this project, an outlier means a value that is much higher or lower than most of the other values in the same feature.

Some features, especially `Insulin`, `BMI`, and `Glucose`, may contain outliers. These values were not removed because they may still represent real medical values, and removing them without a strong reason could remove useful information.

---

## Class Balance
Since this is a classification problem, I checked how many patients were in each class:

- Class `0`: No Diabetes
- Class `1`: Diabetes

This step is important because if one class is much larger than the other, the model may become biased toward predicting the majority class.

```python
df["Outcome"].value_counts()
```

The notebook also includes an outcome distribution chart to visually compare the number of patients with and without diabetes.

```python
sns.countplot(x="Outcome", data=df)
plt.title("Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()
```

---

## Data Visualization
Visualizations were created to better understand the dataset and compare features between diabetes outcomes.

The graphs helped show patterns in features such as:

- Glucose
- BMI
- Age
- Insulin
- Blood Pressure

The notebook includes:

| Visualization | Purpose |
|---|---|
| Outcome Distribution Chart | Shows the number of patients in each class |
| Glucose vs Outcome Boxplot | Compares glucose levels for diabetic and non-diabetic patients |
| BMI vs Outcome Boxplot | Compares BMI values for both outcome groups |
| Age Distribution Chart | Shows how age is distributed between the classes |
| Correlation Heatmap | Shows relationships between features and the target |
| Confusion Matrix Charts | Shows correct and incorrect model predictions |
| Model Accuracy Comparison Chart | Compares Logistic Regression and Random Forest |
| Feature Importance Chart | Shows which features mattered most to Random Forest |

Example visualization code from the notebook:

```python
sns.boxplot(x="Outcome", y="Glucose", data=df)
plt.title("Glucose vs Outcome")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Glucose")
plt.show()
```

Another important chart used was the correlation heatmap:

```python
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

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
- Splitting the data into training and testing sets

Feature scaling was used because models such as Logistic Regression can be affected by features having very different number ranges.

```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
```

The features were scaled using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
```

The data was then split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

---

## Machine Learning Models
Two machine learning models were trained in this project:

- Logistic Regression
- Random Forest

### Logistic Regression
Logistic Regression was used as the baseline model because it is simple, common for binary classification, and easier to explain.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

### Random Forest
Random Forest was also used to compare performance. Random Forest combines many decision trees and can capture more complex patterns in the data.

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

---

## Model Evaluation
The models were evaluated using data they had not seen during training.

The evaluation included:

- Accuracy
- Confusion matrix
- Classification report
- Precision
- Recall
- F1-score

These metrics helped show how well the models predicted both diabetes and non-diabetes outcomes.

Example evaluation code:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

The notebook also includes confusion matrix charts to make the model results easier to understand visually.

```python
log_cm = confusion_matrix(y_test, y_pred)

sns.heatmap(log_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

---

## Model Comparison
Both Logistic Regression and Random Forest were tested using the same training and testing split.

The notebook compares the accuracy scores of both models:

```python
log_acc = accuracy_score(y_test, y_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("Logistic Regression Accuracy:", log_acc)
print("Random Forest Accuracy:", rf_acc)
```

A bar chart was also created to compare model accuracy visually.

```python
model_scores = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [log_acc, rf_acc]
})

sns.barplot(x="Model", y="Accuracy", data=model_scores)
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.show()
```

This comparison helps show which model performed better on the test data.

---

## Feature Importance
Feature importance was reviewed using the Random Forest model. This helps show which columns had the strongest effect on the model’s predictions.

```python
importance = rf_model.feature_importances_
features = X.columns

feat_importance = pd.Series(importance, index=features).sort_values(ascending=False)

feat_importance.plot(kind="bar")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()
```

This connects the machine learning results back to the medical features in the dataset. Features like `Glucose`, `BMI`, and `Age` were expected to be important because they also showed patterns in the earlier visualizations.

---

## Results
The machine learning models were able to make reasonable predictions for diabetes outcomes. The models showed that health features can be used to find patterns related to diabetes.

Some of the most useful indicators were:

- Glucose
- BMI
- Age

Logistic Regression was useful as a simple baseline model, while Random Forest was added as a comparison model because it can capture more complex relationships in the dataset.

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
- Creating a final Kaggle submission file if a separate challenge test set is provided

---

## Conclusion
Overall, this project demonstrates the main steps of a tabular machine learning classification project. The notebook loads the data, explores the features, checks for missing values and outliers, creates visualizations, prepares the data, trains Logistic Regression and Random Forest models, compares their performance, reviews feature importance, and evaluates the results.
