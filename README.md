# Mental Health Machine Learning Project

## 1. Project Overview

This project applies supervised machine learning techniques on a synthetic mental health dataset.

The goal of the project is to analyze demographic, lifestyle, digital behavior, and mental health-related variables and use them to solve two supervised machine learning tasks:

1. **Classification task**: predicting whether an individual shows signs of burnout.
2. **Regression task**: predicting the numerical stress level of an individual.

The project also includes an interactive **Streamlit web application** where the user can:

- view dataset information;
- compare trained machine learning models;
- select a model from a dropdown;
- enter custom input values;
- generate predictions;
- view SHAP explanations for the prediction;
- view learning curves.

The dataset is synthetic and is used for educational machine learning practice. Therefore, the results should be interpreted as a data science exercise, not as a medical, psychological, or clinical diagnosis.

---

## 2. Dataset Description

The dataset used in this project is stored in:

```text
data/mental_health.csv
```

The dataset contains information about individuals and their demographic characteristics, lifestyle habits, digital behavior, and mental health-related indicators.

### Dataset Size

```text
Rows: 2000
Columns: 18
Missing values: 0
```

The dataset does not contain missing values, so no imputation was required.

### Dataset Columns

| Column | Description |
|---|---|
| `Person_ID` | Unique identifier for each individual |
| `Age` | Age of the individual |
| `Gender` | Gender of the individual |
| `Occupation` | Occupation category |
| `Daily_Screen_Time` | Daily screen time |
| `Social_Media_Usage` | Daily social media usage |
| `Night_Usage` | Whether the individual uses digital devices at night |
| `Sleep_Hours` | Average number of sleep hours |
| `Stress_Level` | Numerical stress level from 1 to 10 |
| `Work_Study_Hours` | Number of work or study hours |
| `Physical_Activity` | Physical activity level |
| `Social_Interaction_Score` | Social interaction score |
| `Caffeine_Intake` | Caffeine intake level |
| `Smoking` | Whether the individual smokes |
| `Alcohol` | Whether the individual consumes alcohol |
| `Depression` | Depression indicator |
| `Anxiety` | Anxiety indicator |
| `Burnout` | Burnout indicator |

---

## 3. Project Structure

```text
ProiectML/
│
├── app/
│   └── app.py
│
├── data/
│   └── mental_health.csv
│
├── models/
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   ├── classification_tuned_models.pkl
│   └── regression_tuned_models.pkl
│
├── notebooks/
│   ├── classification_burnout.ipynb
│   └── regression_stress_level.ipynb
│
├── outputs/
│   └── metrics/
│       ├── classification_base_models_results.csv
│       ├── classification_tuned_models_results.csv
│       ├── regression_base_models_results.csv
│       └── regression_tuned_models_results.csv
│
├── requirements.txt
└── README.md
```

### Folder Description

| Folder/File | Description |
|---|---|
| `app/` | Contains the Streamlit application |
| `data/` | Contains the dataset used in the project |
| `models/` | Contains the saved trained models |
| `notebooks/` | Contains the Jupyter notebooks for classification and regression |
| `outputs/metrics/` | Contains CSV files with model evaluation results |
| `requirements.txt` | Contains the Python libraries required to run the project |
| `README.md` | Contains project documentation |

---

## 4. Classification Task: Burnout Prediction

### 4.1 Problem Definition

The classification task aims to predict whether an individual shows signs of burnout.

The target variable is:

```text
Burnout
```

The target contains two classes:

```text
0 = No Burnout
1 = Burnout
```

This is a binary classification problem.

The model uses demographic, lifestyle, digital behavior, and mental health-related variables to predict whether a person belongs to the burnout or no-burnout class.

### 4.2 Input Features

The classification model uses the following input features:

- `Age`
- `Gender`
- `Occupation`
- `Daily_Screen_Time`
- `Social_Media_Usage`
- `Night_Usage`
- `Sleep_Hours`
- `Stress_Level`
- `Work_Study_Hours`
- `Physical_Activity`
- `Social_Interaction_Score`
- `Caffeine_Intake`
- `Smoking`
- `Alcohol`
- `Depression`
- `Anxiety`

The `Person_ID` column was removed because it is only an identifier and does not provide useful predictive information.

### 4.3 Target Distribution

The target variable `Burnout` is almost balanced:

```text
Burnout = 1: 1031 records, approximately 51.55%
Burnout = 0: 969 records, approximately 48.45%
```

This is useful because the models are not strongly biased toward one class.

Since the two classes are close in size, metrics such as accuracy, precision, recall, F1-score, and ROC-AUC can be interpreted more reliably.

### 4.4 Preprocessing

The preprocessing pipeline includes:

1. removing the identifier column `Person_ID`;
2. splitting the dataset into training and testing sets;
3. scaling numerical features using `StandardScaler`;
4. encoding categorical features using `OneHotEncoder`;
5. combining preprocessing steps using `ColumnTransformer`;
6. connecting preprocessing and model training using `Pipeline`.

The dataset was split using a 75/25 train-test split:

```text
Training set: 75%
Testing set: 25%
```

### 4.5 Classification Models Used

The following classification algorithms were trained and evaluated:

1. Naive Bayes
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Support Vector Machine
6. K-Nearest Neighbors
7. XGBoost
8. CatBoost
9. Explainable Boosting Machine

### 4.6 Classification Evaluation Metrics

The classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

These metrics were used because the task is a binary classification problem.

---

## 5. Classification Results

### 5.1 Baseline Classification Models

The baseline models were trained using default or simple initial hyperparameters.

The best baseline model was:

```text
Logistic Regression
Accuracy: 0.618
F1-score: 0.622
ROC-AUC: 0.659
```

### 5.2 Baseline Classification Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.618 | 0.636 | 0.609 | 0.622 | 0.659 |
| Explainable Boosting Machine | 0.618 | 0.637 | 0.605 | 0.620 | 0.661 |
| XGBoost | 0.598 | 0.606 | 0.632 | 0.619 | 0.633 |
| Support Vector Machine | 0.614 | 0.633 | 0.601 | 0.616 | 0.673 |
| Naive Bayes | 0.608 | 0.630 | 0.581 | 0.605 | 0.646 |
| CatBoost | 0.602 | 0.620 | 0.589 | 0.604 | 0.651 |
| Random Forest | 0.600 | 0.624 | 0.566 | 0.593 | 0.640 |
| K-Nearest Neighbors | 0.544 | 0.556 | 0.574 | 0.565 | 0.555 |
| Decision Tree | 0.558 | 0.578 | 0.531 | 0.554 | 0.559 |

### 5.3 Interpretation of Baseline Classification Results

The best baseline model was Logistic Regression.

It achieved an accuracy of approximately 0.62 and an F1-score of approximately 0.62.

Because the target variable is almost balanced, a random classifier would be expected to obtain around 50% accuracy. The obtained results are above this baseline, which means that the models are able to learn useful patterns from the dataset.

However, the performance is moderate. This means that burnout can be predicted to some extent, but the available variables do not perfectly separate the burnout and no-burnout classes.

---

## 6. Classification Hyperparameter Tuning

After evaluating the baseline models, the top five models were selected for hyperparameter tuning.

The selected models were:

1. Logistic Regression
2. Explainable Boosting Machine
3. XGBoost
4. Support Vector Machine
5. Naive Bayes

Hyperparameter tuning was performed using `GridSearchCV`.

The goal of this step was to find better hyperparameter combinations and improve model performance.

### 6.1 Best Tuned Classification Model

After tuning, the best model was:

```text
Explainable Boosting Machine
Accuracy: 0.626
F1-score: 0.631
ROC-AUC: 0.655
```

### 6.2 Tuned Classification Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Explainable Boosting Machine | 0.626 | 0.643 | 0.620 | 0.631 | 0.655 |
| Support Vector Machine | 0.614 | 0.623 | 0.640 | 0.631 | 0.670 |
| Logistic Regression | 0.624 | 0.640 | 0.620 | 0.630 | 0.665 |
| Naive Bayes | 0.608 | 0.630 | 0.581 | 0.605 | 0.646 |
| XGBoost | 0.604 | 0.625 | 0.581 | 0.602 | 0.656 |

### 6.3 Interpretation of Classification Tuning Results

After tuning, the best-performing classification model was the Explainable Boosting Machine.

Compared to the best baseline model, Logistic Regression, the tuned Explainable Boosting Machine achieved a small improvement in F1-score.

The tuned Support Vector Machine and tuned Logistic Regression models obtained very similar F1-scores, both around 0.63. This shows that several models perform comparably on this dataset.

Overall, hyperparameter tuning improved the classification performance slightly, but the improvement was limited. This suggests that the available features contain a moderate predictive signal for burnout, but the two classes are not perfectly separable.

---

## 7. Classification Explainability

SHAP was used to explain the predictions of the top three tuned classification models.

The top three tuned classification models were:

1. Explainable Boosting Machine
2. Support Vector Machine
3. Logistic Regression

### 7.1 Most Important Features for Burnout Prediction

The most relevant features identified by SHAP included:

- `Stress_Level`
- `Work_Study_Hours`
- `Night_Usage`

### 7.2 SHAP Interpretation for Classification

The SHAP analysis showed that higher stress levels tend to increase the prediction toward the burnout class.

Higher work or study hours also tend to push the prediction toward burnout, which suggests that workload may be an important factor in the dataset.

The `Night_Usage` feature had a clear effect. When night usage was active, the SHAP values often pushed the model prediction toward burnout. When night usage was inactive, the feature tended to push the prediction toward no burnout.

Overall, SHAP helped explain both global feature importance and individual predictions.

---

## 8. Classification Learning Curves

Learning curves were generated for the top five tuned classification models.

The purpose of the learning curves was to compare training performance and validation performance as the size of the training set increased.

### 8.1 Learning Curves Interpretation

For Logistic Regression, Support Vector Machine, Naive Bayes, and Explainable Boosting Machine, the training and validation curves became relatively close when more training data was used. This suggests that these models generalize reasonably well and do not show severe overfitting.

For XGBoost, the training score remained higher than the validation score, especially for smaller training set sizes. This indicates a mild tendency toward overfitting.

Overall, the learning curves confirmed that the models achieved moderate predictive performance. The main limitation was not severe overfitting, but rather the moderate predictive signal available in the dataset.

---

## 9. Regression Task: Stress Level Prediction

### 9.1 Problem Definition

The regression task aims to predict the numerical stress level of an individual.

The target variable is:

```text
Stress_Level
```

The target ranges from 1 to 10.

This is a supervised regression problem.

### 9.2 Input Features

For the regression task, the following columns were removed:

- `Person_ID`, because it is only an identifier;
- `Stress_Level`, because it is the target variable;
- `Burnout`, because it is conceptually close to stress and was already used as the classification target.

The regression model uses the remaining demographic, lifestyle, digital behavior, and mental health-related variables.

### 9.3 Regression Preprocessing

The preprocessing pipeline includes:

1. removing non-predictive and target-related columns;
2. splitting the data into training and testing sets;
3. scaling numerical features using `StandardScaler`;
4. encoding categorical features using `OneHotEncoder`;
5. combining all preprocessing steps with `ColumnTransformer`;
6. training models using `Pipeline`.

The data was split using a 75/25 train-test split:

```text
Training set: 75%
Testing set: 25%
```

### 9.4 Regression Models Used

The following regression algorithms were trained and evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regressor
5. K-Nearest Neighbor Regressor
6. Gaussian Process Regressor
7. XGBoost Regressor
8. CatBoost Regressor
9. Explainable Boosting Regressor

### 9.5 Regression Evaluation Metrics

The regression models were evaluated using:

- Mean Squared Error, MSE
- Mean Absolute Error, MAE
- Root Mean Squared Error, RMSE
- R² score

---

## 10. Regression Results

### 10.1 Baseline Regression Models

The best baseline regression model was:

```text
Linear Regression
MSE: 7.601
MAE: 2.332
RMSE: 2.757
R²: 0.105
```

### 10.2 Baseline Regression Results

| Model | MSE | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 7.601 | 2.332 | 2.757 | 0.105 |
| Explainable Boosting Regressor | 7.674 | 2.353 | 2.770 | 0.097 |
| Random Forest Regressor | 7.929 | 2.387 | 2.816 | 0.067 |
| Gaussian Process Regressor | 8.098 | 2.456 | 2.846 | 0.047 |
| CatBoost Regressor | 8.158 | 2.414 | 2.856 | 0.040 |
| Support Vector Regressor | 8.195 | 2.384 | 2.863 | 0.036 |
| K-Nearest Neighbor Regressor | 8.426 | 2.406 | 2.903 | 0.008 |
| XGBoost Regressor | 9.615 | 2.579 | 3.101 | -0.132 |
| Decision Tree Regressor | 15.906 | 3.242 | 3.988 | -0.872 |

### 10.3 Interpretation of Baseline Regression Results

The best baseline regression model was Linear Regression.

It obtained an RMSE of approximately 2.76 and an R² score of approximately 0.105.

The low R² value means that the model explains only a limited part of the variation in stress level. This suggests that the `Stress_Level` variable is difficult to predict accurately using the available features.

The Actual vs Predicted plot showed that predictions were mostly concentrated around the middle of the stress scale. This means that the model had difficulty predicting very low and very high stress levels.

---

## 11. Regression Hyperparameter Tuning

After evaluating the baseline regression models, the top five models were selected for hyperparameter tuning based on RMSE.

The selected models were:

1. Linear Regression
2. Explainable Boosting Regressor
3. Random Forest Regressor
4. Gaussian Process Regressor
5. CatBoost Regressor

Hyperparameter tuning was performed using `GridSearchCV`.

### 11.1 Best Tuned Regression Model

After tuning, the best regression model was:

```text
CatBoost Regressor
MSE: 7.578
MAE: 2.352
RMSE: 2.753
R²: 0.108
```

### 11.2 Tuned Regression Results

| Model | MSE | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| CatBoost Regressor | 7.578 | 2.352 | 2.753 | 0.108 |
| Linear Regression | 7.601 | 2.332 | 2.757 | 0.105 |
| Explainable Boosting Regressor | 7.644 | 2.347 | 2.765 | 0.100 |
| Random Forest Regressor | 7.650 | 2.341 | 2.766 | 0.100 |
| Gaussian Process Regressor | 8.098 | 2.456 | 2.846 | 0.047 |

### 11.3 Interpretation of Regression Tuning Results

After hyperparameter tuning, the best-performing regression model was CatBoost Regressor.

The improvement compared to the best baseline regression model was very small. The RMSE improved only slightly, and the R² score remained low.

This indicates that hyperparameter tuning did not significantly improve the regression task. The limited performance is most likely caused by the weak predictive relationship between the available input variables and the `Stress_Level` target.

The regression task is useful for comparing models and understanding feature influence, but the predictions should not be treated as highly precise.

---

## 12. Regression Learning Curves

Learning curves were generated for the top five tuned regression models.

For regression, the learning curves were evaluated using RMSE. A lower RMSE indicates better performance.

### 12.1 Learning Curves Interpretation

For CatBoost Regressor, Linear Regression, and Explainable Boosting Regressor, the training and validation RMSE values became relatively close as the training set size increased. This suggests that these models do not suffer from severe overfitting, but their predictive performance remains limited.

Random Forest Regressor showed a larger gap between training and validation RMSE, which indicates a tendency toward overfitting.

Gaussian Process Regressor showed a very low training RMSE and a much higher validation RMSE. This indicates strong overfitting, meaning that the model fits the training data very closely but does not generalize well to unseen data.

Overall, the learning curves confirmed that predicting `Stress_Level` is a difficult regression task with the available features.

---

## 13. Regression Explainability

SHAP was used to explain the predictions of the top three tuned regression models.

The top three tuned regression models were:

1. CatBoost Regressor
2. Linear Regression
3. Explainable Boosting Regressor

### 13.1 Most Important Features for Stress Level Prediction

The most relevant features identified by SHAP were:

- `Anxiety`
- `Depression`
- `Daily_Screen_Time`

### 13.2 SHAP Interpretation for Regression

The SHAP analysis showed that anxiety and depression were the most influential features for predicting stress level.

Higher values of `Anxiety` tended to increase the predicted stress level.

A similar pattern appeared for `Depression`. When depression was present or higher, the SHAP values were positive, which means that the feature increased the predicted stress level.

`Daily_Screen_Time` also appeared as an important feature. In the model, lower standardized values of daily screen time tended to have positive SHAP values, while higher standardized values tended to have negative SHAP values. This relationship may reflect the synthetic structure of the dataset.

Overall, SHAP showed that the regression models relied mostly on mental health-related indicators, especially anxiety and depression, when predicting stress level.

---

## 14. Streamlit Application

The project includes an interactive Streamlit application located in:

```text
app/app.py
```

The application contains three main sections:

1. **Project Overview**
2. **Burnout Classification**
3. **Stress Level Regression**

### 14.1 Project Overview Page

The overview page displays:

- general project description;
- dataset overview;
- number of rows;
- number of columns;
- missing values;
- sample dataset records;
- descriptive statistics.

### 14.2 Burnout Classification Page

The classification page allows the user to:

- view the burnout target distribution;
- view baseline classification model results;
- view tuned classification model results;
- select a classification model from a dropdown;
- enter custom input values;
- predict burnout;
- view the predicted burnout probability;
- view SHAP explanation for the current prediction;
- view a learning curve for the selected model.

### 14.3 Stress Level Regression Page

The regression page allows the user to:

- view stress level distribution;
- view correlations with stress level;
- view baseline regression model results;
- view tuned regression model results;
- select a regression model from a dropdown;
- enter custom input values;
- predict stress level;
- view SHAP explanation for the current prediction;
- view a learning curve for the selected model.

---

## 15. Saved Models and Outputs

The trained models are saved in the `models/` folder.

### Saved Classification Models

```text
models/best_classification_model.pkl
models/classification_tuned_models.pkl
```

### Saved Regression Models

```text
models/best_regression_model.pkl
models/regression_tuned_models.pkl
```

### Saved Metrics

The evaluation results are stored in:

```text
outputs/metrics/classification_base_models_results.csv
outputs/metrics/classification_tuned_models_results.csv
outputs/metrics/regression_base_models_results.csv
outputs/metrics/regression_tuned_models_results.csv
```

---

## 16. How to Run the Project

### Step 1: Install Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

If using a local virtual environment on Windows:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 2: Run the Streamlit Application

From the root folder of the project, run:

```bash
streamlit run app/app.py
```

If the `streamlit` command is not recognized, use:

```bash
python -m streamlit run app/app.py
```

or, with the local virtual environment:

```bash
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

---

## 17. Requirements

The main libraries used in this project are:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- catboost
- interpret
- shap
- streamlit
- joblib
- jupyter
- ipykernel

The full list of dependencies is available in:

```text
requirements.txt
```

---

## 18. Final Notes

The project successfully implements two supervised machine learning tasks:

1. Burnout classification
2. Stress level regression

The classification task achieved moderate performance. The best tuned classification model was the Explainable Boosting Machine, with an F1-score of approximately 0.631.

The regression task was more difficult. The best tuned regression model was CatBoost Regressor, with an RMSE of approximately 2.753 and an R² score of approximately 0.108.

The results show that the dataset contains some useful predictive signal, especially for burnout classification, but the models do not achieve very high performance. This is expected because the dataset is synthetic and the relationships between variables are not perfectly separable.

The Streamlit application provides an interactive interface for exploring the dataset, comparing models, generating predictions, and explaining model outputs using SHAP.

Because this project uses a synthetic dataset, the results should be interpreted only as an educational machine learning exercise and not as a real mental health assessment tool.