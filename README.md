# Employee Salary Classification App

This project predicts whether an employee earns more than 50K or less than or equal to 50K annually based on demographic and work-related features. It uses machine learning models trained on the UCI Adult Income dataset and is deployed using Streamlit.

## Features
- Predict salary class (>50K or ≤50K)
- Cleaned and preprocessed dataset
- Trained with multiple ML models and selected the best based on accuracy
- Best model used: Gradient Boosting Classifier
- Web app developed using Streamlit
- Batch prediction support for uploaded CSV files

## Project Structure
employee-salary-classification/
├── app.py                 # Streamlit web app
├── best_model.pkl         # Trained model file
├── adult.csv              # Dataset (UCI Adult Income)
├── generate_best_model.py # Model training script
├── requirements.txt       # Python dependencies

## Dataset Used
- Name: UCI Adult Income Dataset
- Source: https://archive.ics.uci.edu/ml/datasets/adult
- Alternative: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

## How to Run the App

### Step 1: Clone the Repository
git clone https://github.com/YourUsername/employee-salary-classification.git
cd employee-salary-classification

### Step 2: Create Virtual Environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # For Windows
# or
source venv/bin/activate  # For Mac/Linux

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Run the Streamlit App
streamlit run app.py

## Technologies Used
- Python 3.10+
- pandas, numpy
- scikit-learn
- streamlit
- joblib
- matplotlib, seaborn (for EDA)

## Results
- Final model: Gradient Boosting Classifier
- Test accuracy: 86%
- Handles real-time and batch predictions

## Future Scope
- Use XGBoost or ensemble models for higher accuracy
- Dynamic feature encoding based on training mappings
- Deploy as a web service with Docker and CI/CD pipelines

## References
1. UCI Adult Dataset - https://archive.ics.uci.edu/ml/datasets/adult  
2. scikit-learn Documentation - https://scikit-learn.org/  
3. Streamlit Docs - https://docs.streamlit.io/  
4. Joblib Docs - https://joblib.readthedocs.io/  
5. Kaggle Dataset - https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

## Author
Mugilan B
B.S. Abdur Rahman Crescent Institute of Science & Technology
B.Tech Artificial Intelligence & Data Science

## License
This project is intended for academic and educational use only.
