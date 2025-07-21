import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Step 1: Load Data
data = pd.read_csv("adult.csv")

# Step 2: Clean Missing or Ambiguous Values
data['workclass'] = data['workclass'].replace('?', 'Others')
data['occupation'] = data['occupation'].replace('?', 'Others')

# Step 3: Remove non-earning categories
data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]

# Step 4: Remove outliers and less relevant education levels
data = data[(data['age'] >= 17) & (data['age'] <= 75)]
data = data[~data['education'].isin(['1st-4th', '5th-6th', 'Preschool'])]

# Step 5: Drop redundant column
data.drop(columns=['education'], inplace=True)

# Step 6: Encode categorical columns
encoder = LabelEncoder()
for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
    data[col] = encoder.fit_transform(data[col])

# Step 7: Split features and label
X = data.drop(columns=['income'])
Y = data['income']

print("🧠 Training features used in model:", X.columns.tolist())

# Step 8: Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Save scaler for Streamlit app
joblib.dump(scaler, "scaler.pkl")
print("✅ Saved scaler as scaler.pkl")

# Step 10: Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, Y, test_size=0.2, random_state=23, stratify=Y)

# Step 11: Define candidate models
models = {
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC()
}

results = {}

# Step 12: Train and evaluate
for name, model in models.items():
    model.fit(xtrain, ytrain)
    preds = model.predict(xtest)
    acc = accuracy_score(ytest, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Step 13: Save best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
print("✅ Saved as best_model.pkl")
