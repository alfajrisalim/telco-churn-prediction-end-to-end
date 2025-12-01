📊 Telco Customer Churn Prediction — End-to-End Machine Learning Project

An end-to-end Machine Learning project that predicts customer churn using the Telco Customer Churn dataset, complete with data preprocessing, EDA, model training, hyperparameter tuning, SMOTE oversampling, model evaluation, feature importance analysis, and a fully functional Streamlit dashboard for real-time prediction.

🚀 Project Overview

Understanding customer churn is critical for subscription-based businesses.
This project builds and deploys an ML model that identifies customers likely to churn so companies can take proactive retention actions.

This end-to-end project includes:
  -Detailed EDA & insights
  
  -Data cleaning & preprocessing
  
  -Handling missing values
  
  -One-hot encoding & scaling
  
  -Train-test splitting
  
  -Oversampling with SMOTE (to fix imbalanced target)
  
  -Model training with XGBoost
  
  -Hyperparameter tuning optimized for Recall
  
  -Evaluation metrics
  
  -Feature importance visualization
  
  -Streamlit dashboard for real-time prediction
  


🧹 Data Preprocessing

Key steps:
  -Convert TotalCharges to numeric
  
  -Drop missing rows
  
  -Label encode target (Yes → 1, No → 0)
  
  -Split dataset using stratified sampling
  
  -Numerical features → StandardScaler
  
  -Categorical features → OneHotEncoder
  
  -Build preprocessing pipeline using ColumnTransformer

⚖️ Handling Imbalanced Data

The target variable (Churn) is imbalanced.

Therefore, SMOTE oversampling is applied to the training set:
Original:
 0 → 73%
 1 → 27%

After SMOTE:
 0 → 50%
 1 → 50%

This improves the model’s recall for the minority class.

🤖 Model Training
The main model used is XGBoostClassifier with tuned hyperparameters:
  *n_estimators: 100–300
  
  *learning_rate: 0.05–0.1
  
  *max_depth: 3–5
  
  *GridSearchCV is used to optimize Recall, because in churn prediction, missing a churn customer (false negative) is very costly


🖥️ Streamlit Dashboard

A fully interactive dashboard to:
*Preview dataset

*Upload custom dataset

*Visualize churn distribution

*Display feature importance

*Predict churn for individual customers

*Provide actionable insights

Run the dashboard : streamlit run app.py

🧠 Key Learnings & Impact
This project demonstrates real-world Data Science skills:

*Production-ready preprocessing pipelines

*Dealing with imbalanced classification

*Hyperparameter tuning for business-specific metrics

*Model interpretability

*Deploying ML model with Streamlit

*Creating end-to-end ML systems, not just training a model

*Perfect for Data Scientist / ML Engineer portfolios.

🤖 Tech Stack

-Python

-Pandas, NumPy

-Scikit-learn

-XGBoost

-Imbalanced-learn (SMOTE)

-Matplotlib, Seaborn

-Streamlit

-SHAP

-Joblib

👤 Author
Alfajri Salim
Aspiring Data Scientist & Machine Learning Engineer
Focused on building real-world AI applications with strong predictive performance.

