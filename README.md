# Stroke Prediction using Machine Learning
📌 Problem Statement
Stroke is one of the leading causes of serious long-term disability and death worldwide.
Early detection of individuals at high risk of stroke allows doctors to intervene earlier and potentially save lives.

Goal: Build a machine learning model to predict whether a person is likely to suffer a stroke based on their health and lifestyle features.
Dataset Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

🎯 What the Model Predicts
The model predicts whether a patient is at high risk (1) or low risk (0) of stroke,
based on medical, demographic, and lifestyle factors.

🛠 How It Works
1.Data Loading → Import dataset into pandas DataFrame

2.Data Cleaning → Handle missing values (e.g., fill BMI with mean), drop irrelevant columns (id)

3.Encoding → Convert categorical columns to numeric form (Label Encoding / One-Hot Encoding)

4.Feature Selection →

X = All columns except id & stroke

y = stroke column

5.Train-Test Split → Split dataset into 80% train, 20% test

6.Model Training → Random Forest Classifier (can be replaced with XGBoost)

7.Evaluation → Metrics used: Accuracy, Precision, Recall, F1-score, ROC-AUC
 
 📂 Project Structure
 stroke-prediction/
│
├── train_model.py       # Script to train the model  
├── stroke_prediction.ipynb  # Jupyter Notebook version  
├── requirements.txt     # List of dependencies  
├── README.md            # Project documentation  
├── data/
│   └── stroke_data.csv  # Dataset file  
└── images/
    ├── input_sample.png  # Example input data screenshot  
    └── output_sample.png # Example model output screenshot  
🚀 How to Run
1. Clone the repository:
git clone https://github.com/your-username/stroke-prediction.git
cd stroke-prediction
2. Install dependencies:pip install -r requirements.txt
3. Train the model:python train_model.py

📊 Future Improvements
Balance dataset to reduce bias (e.g., SMOTE oversampling).

Hyperparameter tuning for better accuracy.

Try other models like XGBoost, LightGBM.

Deploy using Streamlit or Flask for real-time predictions.



