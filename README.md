
🧑‍⚕️ Disease Prediction using Machine Learning

This project is a Machine Learning-based Disease Prediction System that predicts possible diseases based on given symptoms. It uses multiple classifiers (Naïve Bayes, SVM, and Random Forest) and combines them with a majority voting ensemble for more accurate predictions.

🚀 Features

Handles imbalanced dataset using Random Oversampling.

Trains and evaluates multiple ML models:

Decision Tree

Random Forest

Support Vector Machine (SVM)

Naïve Bayes

Uses Stratified K-Fold Cross Validation for fair evaluation.

Provides confusion matrices and accuracy scores for each model.

Implements an ensemble model (majority voting) to improve performance.

Accepts symptoms as user input and predicts disease with final aggregated result.

📂 Project Workflow
1. Data Preprocessing

Load dataset from CSV (improved_disease_dataset.csv).

Encode target labels (disease names → numbers).

Handle categorical features like gender.

Fill missing values.

Apply Random Oversampling to fix class imbalance.

2. Model Training

Trained multiple classifiers on the dataset:

Naïve Bayes (GaussianNB)

Support Vector Machine (SVM with RBF kernel)

Random Forest

3. Model Evaluation

Used Stratified K-Fold Cross Validation for reliable scoring.

Plotted confusion matrices with Seaborn.

Compared accuracies of individual models.

4. Ensemble (Majority Voting)

Combined predictions from RF, NB, and SVM.

Final prediction = majority vote (mode).

Achieved higher stability and accuracy.

5. Symptom Prediction Function

The function predict_disease(input_symptoms) allows users to input symptoms (comma-separated) and returns:

Predictions from each model.

Final combined disease prediction.

Example:

print(predict_disease("skin_rash,fever,headache"))


Output:

{
  'Random Forest Prediction': 'Chickenpox',
  'Naive Bayes Prediction': 'Measles',
  'SVM Prediction': 'Chickenpox',
  'Final Prediction': 'Chickenpox'
}

📊 Results

Naïve Bayes: Simple and interpretable, but sometimes lower accuracy.

SVM: Works well on non-linear decision boundaries.

Random Forest: Strong performance due to ensemble of trees.

Ensemble (Voting Classifier): Provided best overall results by combining strengths of all models.

🛠️ Tech Stack

Python

Pandas, NumPy → Data preprocessing

Scikit-learn → ML models, cross-validation, oversampling

Matplotlib, Seaborn → Visualization

📌 How to Run

Clone the repository:

git clone https://github.com/yourusername/disease-prediction-ml.git
cd disease-prediction-ml


Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook / Google Colab file.

Test predictions:

predict_disease("cough,fever,headache")

📖 Learnings

Importance of handling class imbalance (oversampling).

Differences between train-test split vs. cross-validation.

How different classifiers (Naïve Bayes, SVM, RF) approach the problem.

The power of ensemble methods in boosting accuracy.

🔮 Future Improvements

Add severity and duration of symptoms for better predictions.

Deploy as a web app (Flask/Streamlit) for interactive use.

Extend dataset for real-world clinical accuracy.

💡 This project is a great demonstration of ML in healthcare applications, showing how different algorithms can be combined to build more reliable systems.
