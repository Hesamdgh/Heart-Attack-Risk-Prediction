#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:41:54 2024

@author: hesamghanbari
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Step 1: Load the training dataset
train_df = pd.read_csv('/Users/hesamghanbari/Library/CloudStorage/OneDrive-Personal/Documents/MBA_materials/Resume_materials/Business_Analytics/Portfolio_projects/Healthcare_analysis/XGBoost model/heart-attack-risk-analysis/train.csv')

# Remove rows with any missing values
train_df = train_df.dropna()

# Convert categorical variables to numeric (assuming gender, family history, etc., are binary or categorical)
train_df['Sex'] = train_df['Sex'].map({'Male': 1, 'Female': 0})  # Example encoding
train_df['Diet'] = train_df['Diet'].map({'Unhealthy': 0, 'Healthy': 1,'Average':0.5})

# Step 2: Split 'Blood Pressure' column into 'Systolic' and 'Diastolic' columns
train_df[['Systolic', 'Diastolic']] = train_df['Blood Pressure'].str.split('/', expand=True)

# Convert 'Systolic' and 'Diastolic' columns to numeric
train_df['Systolic'] = pd.to_numeric(train_df['Systolic'])
train_df['Diastolic'] = pd.to_numeric(train_df['Diastolic'])

# Step 3: Create binary columns based on the conditions
train_df['Systolic_binary'] = train_df['Systolic'].apply(lambda x: 1 if x >= 130 else 0)
train_df['Diastolic_binary'] = train_df['Diastolic'].apply(lambda x: 1 if x >= 80 else 0)

# Step 3: Drop the original 'Blood Pressure' column since we now have Systolic and Diastolic
train_df = train_df.drop(columns=['Blood Pressure'])
train_df = train_df.drop(columns=['Country'])
train_df = train_df.drop(columns=['Continent'])
train_df = train_df.drop(columns=['Hemisphere'])
train_df = train_df.drop(columns=['Systolic'])
train_df = train_df.drop(columns=['Diastolic'])
train_df = train_df.drop(columns=['Diabetes'])
train_df = train_df.drop(columns=['Smoking'])
# Step 2: Automatically exclude 'Patient ID' column
train_df = train_df.drop(columns=['Patient ID'])

# Step 3: Convert categorical variables automatically using one-hot encoding
train_df = pd.get_dummies(train_df, drop_first=True)  # This will handle all categorical variables automatically

# Step 4: Features and Target Variables
X_train = train_df.drop(columns=['Heart Attack Risk'])  # Drop target column from features
y_train = train_df['Heart Attack Risk']  # Target variable

# Step 3: Apply SMOTE to handle class imbalance in the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Scale the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Step 5: Initialize XGBoost Classifier
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=10000, random_state=42, eval_metric='logloss')

# Step 6: Train the model
xgb_model.fit(X_resampled_scaled, y_resampled)

# Step 7: Load the testing dataset (including Patient ID)
test_df = pd.read_csv('/Users/hesamghanbari/Library/CloudStorage/OneDrive-Personal/Documents/MBA_materials/Resume_materials/Business_Analytics/Portfolio_projects/Healthcare_analysis/XGBoost model/heart-attack-risk-analysis/test.csv')

# Remove rows with any missing values in the test set
test_df = test_df.dropna()

# Extract the Patient ID
patient_ids = test_df['Patient ID']

# Convert categorical variables to numeric (assuming gender, family history, etc., are binary or categorical)
test_df['Sex'] = test_df['Sex'].map({'Male': 1, 'Female': 0})  # Example encoding
test_df['Diet'] = test_df['Diet'].map({'Unhealthy': 0, 'Healthy': 1,'Average':0.5})

# Step 2: Split 'Blood Pressure' column into 'Systolic' and 'Diastolic' columns
test_df[['Systolic', 'Diastolic']] = test_df['Blood Pressure'].str.split('/', expand=True)

# Convert 'Systolic' and 'Diastolic' columns to numeric
test_df['Systolic'] = pd.to_numeric(test_df['Systolic'])
test_df['Diastolic'] = pd.to_numeric(test_df['Diastolic'])

# Step 3: Create binary columns based on the conditions
test_df['Systolic_binary'] = test_df['Systolic'].apply(lambda x: 1 if x >= 130 else 0)
test_df['Diastolic_binary'] = test_df['Diastolic'].apply(lambda x: 1 if x >= 80 else 0)

# Step 3: Drop the original 'Blood Pressure' column since we now have Systolic and Diastolic
test_df = test_df.drop(columns=['Blood Pressure'])
test_df = test_df.drop(columns=['Country'])
test_df = test_df.drop(columns=['Continent'])
test_df = test_df.drop(columns=['Hemisphere'])
test_df = test_df.drop(columns=['Systolic'])
test_df = test_df.drop(columns=['Diastolic'])
test_df = test_df.drop(columns=['Diabetes'])
test_df = test_df.drop(columns=['Smoking'])
test_df = test_df.drop(columns=['Patient ID'])

# Step 3: Convert categorical variables automatically using one-hot encoding
test_df = pd.get_dummies(test_df, drop_first=True)  # This will handle all categorical variables automatically

X_test = test_df

# Step 8: Scale the test data
X_test_scaled = scaler.transform(X_test)

# Step 9: Make predictions on the test set
test_predictions = xgb_model.predict(X_test_scaled)

# Step 10: Create a DataFrame with the Patient ID and prediction results
prediction_results = pd.DataFrame({
    'Patient ID': patient_ids,
    'Heart Attack Risk Prediction': test_predictions
})

# Step 11: Save the results to a new CSV file
prediction_results.to_csv('heart_attack_risk_predictions_with_patient_id_3.csv', index=False)

# Optionally, print out the predictions for verification
print(prediction_results.head())