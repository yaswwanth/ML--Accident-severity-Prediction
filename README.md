# Road Accident Severity Prediction

This project aims to predict the severity of road accidents using machine learning classification algorithms. The dataset contains various features related to accidents, such as weather conditions, time of the accident, and road surface type. Different models like XGBoost, Random Forest, SVM (Linear & RBF), and KNN are applied to predict the severity of road accidents.


## Project Overview

The project is focused on predicting the severity of road accidents using machine learning models. The goal is to classify accident data into different severity levels based on various features. The dataset is preprocessed using outlier detection techniques like Interquartile Range (IQR) and Z-Score to improve the accuracy of predictions.

## Dataset

The dataset used in this project contains the following features:
- **Weather conditions**: Weather type during the accident.
- **Time of the accident**: Time when the accident occurred.
- **Road surface type**: Type of road surface during the accident.
- Additional features related to road conditions and accident details.

Outliers in the dataset are handled using IQR and Z-Score techniques to ensure the accuracy of predictions.

## Preprocessing Steps

The following preprocessing steps are applied to the dataset before training the models:

1. **Data Cleaning**: Missing values are handled, and data consistency is ensured.
2. **Feature Scaling**: Models like SVM and KNN require standardization or normalization of features.
3. **Outlier Detection**: Outliers are identified and removed using IQR and Z-Score methods.
4. **Dimensionality Reduction**: PCA (Principal Component Analysis) is used for dimensionality reduction to improve model efficiency.
5. **Class Balancing**: The dataset suffers from class imbalance, which is handled through:
   - **SMOTE (Synthetic Minority Over-sampling Technique)** for oversampling the minority class.
   - **Weighted classification** to give more importance to minority classes.

## Data Transformation

Data transformation techniques applied to this dataset include:

1. **Encoding Categorical Features**: Categorical variables (e.g., weather conditions, road surface type) are encoded using techniques like One-Hot Encoding or Label Encoding.
2. **Feature Engineering**: New features, if applicable, are created based on existing ones (e.g., extracting hour or day from the accident time).
3. **Log Transformation**: Some numerical features (if skewed) are transformed using a logarithmic scale to reduce skewness and make the data more normally distributed.
4. **Outlier Handling**: Outliers are detected and removed using IQR and Z-Score methods to improve model performance.
5. **Principal Component Analysis (PCA)**: PCA is applied to reduce the dimensionality of the data while retaining as much variance as possible. This helps to improve model performance and reduce overfitting, especially in high-dimensional datasets.

## Machine Learning Models

The following machine learning models are implemented and tested:

1. **XGBoost**: A powerful gradient boosting model.
2. **Random Forest**: A versatile ensemble learning model.
3. **Support Vector Machine (SVM)**: Both Linear and RBF (Radial Basis Function) kernels are used for classification.
4. **K-Nearest Neighbors (KNN)**: A simple, instance-based learning model.

## Model Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy**: The overall accuracy of the model.
- **Precision, Recall, F1-score**: These metrics are computed for each class to evaluate the performance of the models in predicting different accident severities.
- **Confusion Matrix**: To visualize the classification results and errors.

## Technologies and Python Libraries Used

This project uses the following Python libraries and technologies:

1. **Pandas**: For data manipulation and cleaning.
2. **NumPy**: For numerical operations and transformations.
3. **Scikit-learn**: For implementing machine learning algorithms (Random Forest, SVM, KNN) and preprocessing tasks (StandardScaler, PCA, etc.).
4. **XGBoost**: For implementing the XGBoost model.
5. **imbalanced-learn (SMOTE)**: For handling class imbalance using the Synthetic Minority Over-sampling Technique (SMOTE).
6. **Matplotlib** and **Seaborn**: For data visualization, plotting confusion matrix, and evaluating model performance.
7. **SciPy**: For statistical methods like Z-Score.
8. **Statsmodels**: For advanced statistical methods (if applicable).


