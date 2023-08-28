import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the dataset and preprocess it
data = pd.read_csv('C:/Users/ANN/Desktop/Machine Learning for Groundwater Prediction and Analysis/water_potability.csv')
# More detailed preprocessing steps here (handling missing values, outliers, scaling, etc.)

# Handle Missing Values with Mean Imputation
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)
data = pd.DataFrame(data_imputed, columns=data.columns)

# Split the data into features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Implementation and Hyperparameter Tuning
# Random Forest hyperparameter tuning
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=rf_param_grid, cv=5)
rf_grid.fit(X_train_scaled, y_train)
best_rf_model = rf_grid.best_estimator_

# SVM hyperparameter tuning
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'class_weight': ['balanced', None]
}
svm_grid = GridSearchCV(SVC(random_state=42), param_grid=svm_param_grid, cv=5)
svm_grid.fit(X_train_scaled, y_train)
best_svm_model = svm_grid.best_estimator_

# Ensemble Model
ensemble_model = VotingClassifier(estimators=[
    ('rf', best_rf_model),
    ('svm', best_svm_model)
], voting='hard')
ensemble_model.fit(X_train_scaled, y_train)
ensemble_preds = ensemble_model.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)

# Model Evaluation
best_rf_preds = best_rf_model.predict(X_test_scaled)
best_svm_preds = best_svm_model.predict(X_test_scaled)

best_rf_accuracy = accuracy_score(y_test, best_rf_preds)
best_svm_accuracy = accuracy_score(y_test, best_svm_preds)

# Visualization
models = ['Random Forest', 'SVM', 'Ensemble']
accuracies = [best_rf_accuracy, best_svm_accuracy, ensemble_accuracy]

plt.bar(models, accuracies)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()

# Print classification reports
print("Random Forest Classification Report:\n", classification_report(y_test, best_rf_preds))
print("SVM Classification Report:\n", classification_report(y_test, best_svm_preds))
print("Ensemble Classification Report:\n", classification_report(y_test, ensemble_preds))