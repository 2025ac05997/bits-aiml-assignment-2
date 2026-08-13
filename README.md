# ML Assignment 2 – Classification Models using Streamlit

---

## a. Problem Statement  

The objective of this assignment is to design and implement an end-to-end machine learning classification system using a real-world dataset. The task involves training multiple classification models on the same dataset, evaluating them using standard performance metrics, and deploying the models through an interactive Streamlit web application.

The application allows users to upload a dataset, select a classification model, and view detailed evaluation results including accuracy, AUC, precision, recall, F1 score, Matthews Correlation Coefficient (MCC), classification report, and confusion matrix. This assignment demonstrates the complete machine learning workflow: data preprocessing, model training, evaluation, UI design, and deployment.

---

## b. Dataset Description

The dataset used for this assignment is the **Absenteeism at Work** dataset from the **UCI Machine Learning Repository**.

- **Source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/445/absenteeism+at+work  
- **Number of instances:** 740  
- **Number of features:** 19  
- **Original target variable:** *Absenteeism time in hours*  
- **Data format:** CSV

### Target Variable Engineering  

Since the original target is a continuous variable, it was converted into a **binary classification problem** as follows:

- **Class 0:** Absenteeism time **< 8 hours**  
- **Class 1:** Absenteeism time **≥ 8 hours**

This transformation enables the application of standard classification algorithms while maintaining interpretability in the context of employee absenteeism analysis.

---

## c. Models Used and Evaluation Metrics

The following five classification models were implemented using the same dataset and preprocessing pipeline:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  

### Evaluation Metrics Used  

Each model was evaluated using the following metrics:

- **Accuracy** – Overall correctness of predictions  
- **AUC (Area Under ROC Curve)** – Ability to distinguish between classes  
- **Precision** – Correctness of positive predictions  
- **Recall** – Ability to identify actual positive cases  
- **F1 Score** – Harmonic mean of precision and recall  
- **Matthews Correlation Coefficient (MCC)** – Balanced measure accounting for all confusion matrix values  

---

### Comparison Table of All Models  

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8 | 0.8582 | 0.745 | 0.69 | 0.7165 | 0.5633 |
| Decision Tree | 0.4378 | 0.3758 | 0.1749 | 0.1439 | 0.1579 | -0.2608 |
| kNN | 0.8378 | 0.9122 | 0.8186 | 0.7159 | 0.7638 | 0.6445 |
| Naive Bayes | 0.4203 | 0.7891 | 0.3871 | 1.0 | 0.5582 | 0.1817 |
| Random Forest (Ensemble) | 0.4068 | 0.3579 | 0.1613 | 0.1476 | 0.1541 | -0.302 |

*(Metric values are computed dynamically in the Streamlit application.)*

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Demonstrates strong baseline performance with good AUC and MCC, indicating effective discrimination on scaled numerical features. |
| Decision Tree | Shows poor predictive performance with low AUC and negative MCC, indicating limited generalization on this dataset. |
| kNN | Achieves the best overall performance across all evaluation metrics, benefiting significantly from feature scaling and local neighborhood learning. |
| Naive Bayes | Exhibits very high recall but low precision, suggesting a strong bias toward predicting the positive class due to feature independence assumptions. |
| Random Forest (Ensemble) | Underperforms in this implementation, with negative MCC indicating poor generalization despite ensemble averaging. |

---

## Streamlit Application Description  

The Streamlit web application provides an interactive interface with the following features:

- Dataset upload option (CSV)  
- Model selection dropdown  
- Display of evaluation metrics  
- Detailed classification report  
- Confusion matrix visualization  

The application dynamically loads models from the `model/` directory, ensuring modular design and separation between the UI and model logic.

---

## Deployment  

The application is deployed using **Streamlit Community Cloud**. The deployed app opens an interactive frontend that allows users to upload data, select models, and analyze classification performance in real time.

---

## Summary  

This assignment successfully demonstrates:

- Implementation of multiple machine learning classification models  
- Comparative evaluation using standard metrics  
- Clean modular code structure  
- Deployment of an interactive machine learning application using Streamlit  

The solution strictly follows the assignment requirements and reflects best practices in machine learning development and deployment.
