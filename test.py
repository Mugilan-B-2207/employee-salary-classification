import joblib
import pandas as pd

model = joblib.load("best_model.pkl")

# Try different test inputs
test_input = pd.DataFrame([{
    'age': 25,
    'workclass': 2,
    'fnlwgt': 120000,
    'educational-num': 8,
    'marital-status': 4,
    'occupation': 6,
    'relationship': 1,
    'race': 3,
    'gender': 1,
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 20,
    'native-country': 2
}])

print("Prediction:", model.predict(test_input)[0])
