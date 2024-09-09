import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from flask import Flask, request, render_template

# Load and preprocess the dataset
df = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\Mini2\\heart_disease_modelcsv.csv")

# Rename the target column
df.rename(columns={"Heart Disease": "target"}, inplace=True)

# Convert the target column to categorical codes if needed
df["target"] = df["target"].astype("category").cat.codes

# Handle missing values by filling with mean (if any)
df.fillna(df.mean(), inplace=True)

# Split dataset into features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')

# Create the Flask web application
app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form and process it
        data = []
        for key, value in request.form.items():
            if key == 'exercise_angina':
                data.append(1 if value == 'yes' else 0)  # Convert exercise_angina to binary
            else:
                data.append(float(value))  # Convert other values to float
        
        # Convert to numpy array
        input_data = np.array([data])
        
        # Predict the outcome
        prediction = model.predict(input_data)
        output = 'Has Heart Disease' if prediction[0] == 1 else 'Does Not Have Heart Disease'
        
        return render_template('result.html', prediction_text=f'The person {output}')
    except ValueError as e:
        # Handle invalid input
        return render_template('result.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
