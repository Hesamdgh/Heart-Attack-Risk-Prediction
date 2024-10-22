#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:44:51 2024

@author: hesamghanbari
"""

# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the training dataset
df = pd.read_csv('/Users/hesamghanbari/Library/CloudStorage/OneDrive-Personal/Documents/MBA_materials/Resume_materials/Business_Analytics/Portfolio_projects/Healthcare_analysis/XGBoost model/heart-attack-risk-analysis/train.csv')

# Remove rows with any missing values
df = df.dropna()

# Convert categorical variables to numeric (assuming gender, family history, etc., are binary or categorical)
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})  # Example encoding
df['Diet'] = df['Diet'].map({'Unhealthy': 0, 'Healthy': 1,'Average':0.5})

# Step 2: Split 'Blood Pressure' column into 'Systolic' and 'Diastolic' columns
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)

# Convert 'Systolic' and 'Diastolic' columns to numeric
df['Systolic'] = pd.to_numeric(df['Systolic'])
df['Diastolic'] = pd.to_numeric(df['Diastolic'])

# Step 3: Create binary columns based on the conditions
df['Systolic_binary'] = df['Systolic'].apply(lambda x: 1 if x >= 130 else 0)
df['Diastolic_binary'] = df['Diastolic'].apply(lambda x: 1 if x >= 80 else 0)

# Step 3: Drop the original 'Blood Pressure' column since we now have Systolic and Diastolic
df = df.drop(columns=['Blood Pressure'])
df = df.drop(columns=['Country'])
df = df.drop(columns=['Continent'])
df = df.drop(columns=['Hemisphere'])
df = df.drop(columns=['Systolic'])
df = df.drop(columns=['Diastolic'])
df = df.drop(columns=['Sex'])
df = df.drop(columns=['Diet'])
df = df.drop(columns=['Systolic_binary'])
df = df.drop(columns=['Diastolic_binary'])
# Step 2: Automatically exclude 'Patient ID' column
df = df.drop(columns=['Patient ID'])

# Step 3: Convert categorical variables automatically using one-hot encoding
df = pd.get_dummies(df, drop_first=True)  # This will handle all categorical variables automatically

# Step 4: Features and Target Variables
X = df.drop(columns=['Heart Attack Risk'])  # Drop target column from features
y = df['Heart Attack Risk']  # Target variable

# Step 5: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Step 8: Assess feature importance
feature_importances = rf_model.feature_importances_

# Step 9: Create a DataFrame for better visualization
feature_names = X.columns  # Use all column names except 'Patient ID' and 'Heart Attack Risk'
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Step 8: Use RFECV to determine the optimal number of features
rfecv = RFECV(estimator=rf_model, step=1, cv=5, scoring='accuracy', n_jobs=-1)
rfecv.fit(X_scaled, y)

# Step 9: Plot the number of selected features vs cross-validation scores
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (accuracy)")
plt.title("Optimum Number of Features")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()

# Step 10: Print the optimal number of features
print(f"Optimal number of features: {rfecv.n_features_}")

# Step 10: Visualize feature importances using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance for Heart Attack Risk Prediction (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Print the importance of each feature
print(importance_df)