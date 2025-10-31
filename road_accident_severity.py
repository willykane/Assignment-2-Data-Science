# ============================
# AUTHOR : WILLYCE OJWANG
# REG NO : BSE-05-0044/2024
# GROUP 3
# ============================

# Road Acciedent Severity Prediction Project

#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = {
    'Speed':[50, 70, 30, 100, 60, 90, 40, 110],
    'Weather_Condition': [1, 2, 1, 3, 2, 1, 2, 3],
    'Road_Surface': [1, 2, 1, 3, 2, 1, 2, 3],
    'Light_Condition': [1, 2, 1, 3, 2, 1, 2, 3],
    'Vehicle_Age': [2, 5, 1, 8, 3, 6, 2, 10],
    'Accident_Severity': [2, 4, 1, 5, 3, 4, 2, 5],
}

df = pd.DataFrame(data)
print("== Dataset Loaded ==")
print(df)

#Define features and target variable
X = df[['Speed', 'Weather_Condition', 'Road_Surface', 'Light_Condition', 'Vehicle_Age']]
y = df['Accident_Severity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print("\n=== Model Evaluation ===")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

# Save the trained model
joblib.dump(model, 'accident_severity_model.pkl')
print("\nModel saved successfully as 'accident_severity_model.pkl'")

# Use the model for prediction
loaded_model = joblib.load('accident_severity_model.pkl')
new_data = pd.DataFrame({
    'Speed': [85],
    'Weather_Condition': [2],
    'Road_Surface': [2],
    'Light_Condition': [2],
    'Vehicle_Age': [5]
})

prediction = loaded_model.predict(new_data)
print("\n=== Prediction for New Data ===")
print(new_data)
print(f"Predicted Accident Severity: {prediction[0]:.2f}")
