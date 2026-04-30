import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Mental Health ML App",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "mental_health.csv"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFICATION_MODELS_PATH = MODELS_DIR / "classification_tuned_models.pkl"
REGRESSION_MODELS_PATH = MODELS_DIR / "regression_tuned_models.pkl"

BEST_CLASSIFICATION_MODEL_PATH = MODELS_DIR / "best_classification_model.pkl"
BEST_REGRESSION_MODEL_PATH = MODELS_DIR / "best_regression_model.pkl"

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
CLASSIFICATION_BASE_METRICS_PATH = METRICS_DIR / "classification_base_models_results.csv"
CLASSIFICATION_TUNED_METRICS_PATH = METRICS_DIR / "classification_tuned_models_results.csv"
REGRESSION_BASE_METRICS_PATH = METRICS_DIR / "regression_base_models_results.csv"
REGRESSION_TUNED_METRICS_PATH = METRICS_DIR / "regression_tuned_models_results.csv"


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_classification_models():
    if CLASSIFICATION_MODELS_PATH.exists():
        return joblib.load(CLASSIFICATION_MODELS_PATH)

    if BEST_CLASSIFICATION_MODEL_PATH.exists():
        return {"Best Classification Model": joblib.load(BEST_CLASSIFICATION_MODEL_PATH)}

    return {}


@st.cache_resource
def load_regression_models():
    if REGRESSION_MODELS_PATH.exists():
        return joblib.load(REGRESSION_MODELS_PATH)

    if BEST_REGRESSION_MODEL_PATH.exists():
        return {"Best Regression Model": joblib.load(BEST_REGRESSION_MODEL_PATH)}

    return {}


@st.cache_data
def load_metrics(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def get_unique_values(df, column):
    return sorted(df[column].dropna().unique().tolist())


def binary_label(value):
    return "Yes" if int(value) == 1 else "No"


def display_dataset_overview(df):
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing values", int(df.isnull().sum().sum()))

    st.dataframe(df.head(10), use_container_width=True)


def plot_target_distribution_classification(df):
    st.subheader("Burnout Distribution")

    counts = df["Burnout"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["No Burnout", "Burnout"], counts.values)
    ax.set_title("Distribution of Burnout Classes")
    ax.set_xlabel("Burnout")
    ax.set_ylabel("Number of Records")
    st.pyplot(fig)


def plot_stress_distribution(df):
    st.subheader("Stress Level Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["Stress_Level"], bins=10, edgecolor="black")
    ax.set_title("Distribution of Stress Level")
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


def plot_correlation_with_stress(df):
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if "Person_ID" in numeric_columns:
        numeric_columns.remove("Person_ID")

    correlations = df[numeric_columns].corr()["Stress_Level"].drop("Stress_Level").sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    correlations.plot(kind="barh", ax=ax)
    ax.set_title("Correlation of Numerical Features with Stress Level")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


def get_model_params(model_pipeline):
    model = model_pipeline.named_steps["model"]
    params = model.get_params()

    simple_params = {}
    for key, value in params.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            simple_params[key] = value

    return simple_params


def get_processed_data(pipeline, X_background, X_input):
    preprocessor = pipeline.named_steps["preprocessor"]

    X_background_processed = preprocessor.transform(X_background)
    X_input_processed = preprocessor.transform(X_input)

    feature_names = preprocessor.get_feature_names_out()
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    X_background_processed = pd.DataFrame(
        X_background_processed,
        columns=feature_names
    )

    X_input_processed = pd.DataFrame(
        X_input_processed,
        columns=feature_names
    )

    return X_background_processed, X_input_processed, feature_names


def show_classification_shap(model_pipeline, X_train, input_df):
    st.subheader("SHAP Explanation for Current Prediction")

    try:
        model = model_pipeline.named_steps["model"]

        background = X_train.sample(
            n=min(80, len(X_train)),
            random_state=42
        )

        background_processed, input_processed, feature_names = get_processed_data(
            model_pipeline,
            background,
            input_df
        )

        masker = shap.maskers.Independent(background_processed)

        explainer = shap.PermutationExplainer(
            model.predict_proba,
            masker,
            feature_names=feature_names
        )

        shap_values = explainer(
            input_processed,
            max_evals=2 * input_processed.shape[1] + 1
        )

        if len(shap_values.values.shape) == 3:
            shap_values_class_1 = shap.Explanation(
                values=shap_values.values[:, :, 1],
                base_values=shap_values.base_values[:, 1],
                data=input_processed.values,
                feature_names=feature_names
            )
        else:
            shap_values_class_1 = shap_values

        fig = plt.figure()
        shap.plots.waterfall(shap_values_class_1[0], max_display=10, show=False)
        st.pyplot(fig)

        mean_abs = np.abs(shap_values_class_1.values[0])
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Absolute SHAP Value": mean_abs
        }).sort_values(by="Absolute SHAP Value", ascending=False).head(10)

        st.write("Top features influencing this prediction:")
        st.dataframe(importance_df, use_container_width=True)

    except Exception as error:
        st.warning(f"SHAP explanation could not be generated: {error}")


def show_regression_shap(model_pipeline, X_train, input_df):
    st.subheader("SHAP Explanation for Current Prediction")

    try:
        model = model_pipeline.named_steps["model"]

        background = X_train.sample(
            n=min(80, len(X_train)),
            random_state=42
        )

        background_processed, input_processed, feature_names = get_processed_data(
            model_pipeline,
            background,
            input_df
        )

        masker = shap.maskers.Independent(background_processed)

        explainer = shap.PermutationExplainer(
            model.predict,
            masker,
            feature_names=feature_names
        )

        shap_values = explainer(
            input_processed,
            max_evals=2 * input_processed.shape[1] + 1
        )

        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

        mean_abs = np.abs(shap_values.values[0])
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Absolute SHAP Value": mean_abs
        }).sort_values(by="Absolute SHAP Value", ascending=False).head(10)

        st.write("Top features influencing this prediction:")
        st.dataframe(importance_df, use_container_width=True)

    except Exception as error:
        st.warning(f"SHAP explanation could not be generated: {error}")


def show_classification_learning_curve(model_pipeline, X, y):
    st.subheader("Learning Curve")

    with st.spinner("Computing learning curve..."):
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=model_pipeline,
            X=X,
            y=y,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=3,
            scoring="f1",
            n_jobs=-1
        )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, marker="o", label="Training F1")
    ax.plot(train_sizes, validation_mean, marker="o", label="Validation F1")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def show_regression_learning_curve(model_pipeline, X, y):
    st.subheader("Learning Curve")

    with st.spinner("Computing learning curve..."):
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=model_pipeline,
            X=X,
            y=y,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

    train_rmse = -train_scores.mean(axis=1)
    validation_rmse = -validation_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
    ax.plot(train_sizes, validation_rmse, marker="o", label="Validation RMSE")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# -----------------------------
# Load data and models
# -----------------------------
df = load_data()

classification_models = load_classification_models()
regression_models = load_regression_models()

classification_base_metrics = load_metrics(CLASSIFICATION_BASE_METRICS_PATH)
classification_tuned_metrics = load_metrics(CLASSIFICATION_TUNED_METRICS_PATH)
regression_base_metrics = load_metrics(REGRESSION_BASE_METRICS_PATH)
regression_tuned_metrics = load_metrics(REGRESSION_TUNED_METRICS_PATH)


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Mental Health ML App")

page = st.sidebar.radio(
    "Choose page",
    [
        "Project Overview",
        "Burnout Classification",
        "Stress Level Regression"
    ]
)


# -----------------------------
# Project Overview
# -----------------------------
if page == "Project Overview":
    st.title("Mental Health Machine Learning Project")

    st.write(
        """
        This Streamlit application presents two supervised machine learning tasks
        built on the mental health dataset.

        The first task is a binary classification problem, where the model predicts
        whether an individual shows signs of burnout.

        The second task is a regression problem, where the model predicts the
        numerical stress level of an individual.
        """
    )

    display_dataset_overview(df)

    st.subheader("Available Columns")
    st.write(df.columns.tolist())

    st.subheader("Dataset Description")
    st.dataframe(df.describe(), use_container_width=True)


# -----------------------------
# Burnout Classification Page
# -----------------------------
elif page == "Burnout Classification":
    st.title("Burnout Classification")

    st.write(
        """
        This page predicts whether an individual is likely to show signs of burnout.

        The target variable is `Burnout`:
        - `0` means No Burnout
        - `1` means Burnout
        """
    )

    display_dataset_overview(df)
    plot_target_distribution_classification(df)

    st.subheader("Model Metrics")

    if not classification_base_metrics.empty:
        st.write("Base classification models:")
        st.dataframe(classification_base_metrics, use_container_width=True)

    if not classification_tuned_metrics.empty:
        st.write("Tuned classification models:")
        st.dataframe(classification_tuned_metrics, use_container_width=True)

    if not classification_models:
        st.error("No classification models found in the models folder.")
        st.stop()

    selected_model_name = st.selectbox(
        "Select classification model",
        list(classification_models.keys())
    )

    selected_model = classification_models[selected_model_name]

    st.subheader("Selected Model Hyperparameters")
    st.json(get_model_params(selected_model))

    st.subheader("Input Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 16, 100, int(df["Age"].median()))
        gender = st.selectbox("Gender", get_unique_values(df, "Gender"))
        occupation = st.selectbox("Occupation", get_unique_values(df, "Occupation"))
        daily_screen_time = st.number_input(
            "Daily Screen Time",
            float(df["Daily_Screen_Time"].min()),
            float(df["Daily_Screen_Time"].max()),
            float(df["Daily_Screen_Time"].median())
        )
        social_media_usage = st.number_input(
            "Social Media Usage",
            float(df["Social_Media_Usage"].min()),
            float(df["Social_Media_Usage"].max()),
            float(df["Social_Media_Usage"].median())
        )

    with col2:
        night_usage = st.selectbox("Night Usage", [0, 1], format_func=binary_label)
        sleep_hours = st.number_input(
            "Sleep Hours",
            float(df["Sleep_Hours"].min()),
            float(df["Sleep_Hours"].max()),
            float(df["Sleep_Hours"].median())
        )
        stress_level = st.number_input(
            "Stress Level",
            int(df["Stress_Level"].min()),
            int(df["Stress_Level"].max()),
            int(df["Stress_Level"].median())
        )
        work_study_hours = st.number_input(
            "Work/Study Hours",
            float(df["Work_Study_Hours"].min()),
            float(df["Work_Study_Hours"].max()),
            float(df["Work_Study_Hours"].median())
        )
        physical_activity = st.selectbox(
            "Physical Activity",
            get_unique_values(df, "Physical_Activity")
        )

    with col3:
        social_interaction_score = st.number_input(
            "Social Interaction Score",
            int(df["Social_Interaction_Score"].min()),
            int(df["Social_Interaction_Score"].max()),
            int(df["Social_Interaction_Score"].median())
        )
        caffeine_intake = st.number_input(
            "Caffeine Intake",
            int(df["Caffeine_Intake"].min()),
            int(df["Caffeine_Intake"].max()),
            int(df["Caffeine_Intake"].median())
        )
        smoking = st.selectbox("Smoking", [0, 1], format_func=binary_label)
        alcohol = st.selectbox("Alcohol", [0, 1], format_func=binary_label)
        depression = st.selectbox("Depression", [0, 1], format_func=binary_label)
        anxiety = st.selectbox("Anxiety", [0, 1], format_func=binary_label)

    input_classification = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Daily_Screen_Time": daily_screen_time,
        "Social_Media_Usage": social_media_usage,
        "Night_Usage": night_usage,
        "Sleep_Hours": sleep_hours,
        "Stress_Level": stress_level,
        "Work_Study_Hours": work_study_hours,
        "Physical_Activity": physical_activity,
        "Social_Interaction_Score": social_interaction_score,
        "Caffeine_Intake": caffeine_intake,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "Depression": depression,
        "Anxiety": anxiety
    }])

    st.write("Input row:")
    st.dataframe(input_classification, use_container_width=True)

    if st.button("Predict Burnout"):
        prediction = selected_model.predict(input_classification)[0]

        st.subheader("Prediction Result")

        if int(prediction) == 1:
            st.error("Prediction: Burnout")
        else:
            st.success("Prediction: No Burnout")

        if hasattr(selected_model.named_steps["model"], "predict_proba"):
            probability = selected_model.predict_proba(input_classification)[0][1]
            st.metric("Predicted probability of burnout", f"{probability:.3f}")

        X_classification = df.drop(columns=["Person_ID", "Burnout"])
        y_classification = df["Burnout"]

        show_classification_shap(
            selected_model,
            X_classification,
            input_classification
        )

    with st.expander("Show learning curve for selected model"):
        X_classification = df.drop(columns=["Person_ID", "Burnout"])
        y_classification = df["Burnout"]

        if st.button("Compute classification learning curve"):
            show_classification_learning_curve(
                selected_model,
                X_classification,
                y_classification
            )


# -----------------------------
# Stress Level Regression Page
# -----------------------------
elif page == "Stress Level Regression":
    st.title("Stress Level Regression")

    st.write(
        """
        This page predicts the numerical `Stress_Level` value.

        The target variable is `Stress_Level`, which ranges from 1 to 10.
        """
    )

    display_dataset_overview(df)
    plot_stress_distribution(df)
    plot_correlation_with_stress(df)

    st.subheader("Model Metrics")

    if not regression_base_metrics.empty:
        st.write("Base regression models:")
        st.dataframe(regression_base_metrics, use_container_width=True)

    if not regression_tuned_metrics.empty:
        st.write("Tuned regression models:")
        st.dataframe(regression_tuned_metrics, use_container_width=True)

    if not regression_models:
        st.error("No regression models found in the models folder.")
        st.stop()

    selected_regression_model_name = st.selectbox(
        "Select regression model",
        list(regression_models.keys())
    )

    selected_regression_model = regression_models[selected_regression_model_name]

    st.subheader("Selected Model Hyperparameters")
    st.json(get_model_params(selected_regression_model))

    st.subheader("Input Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 16, 100, int(df["Age"].median()), key="reg_age")
        gender = st.selectbox("Gender", get_unique_values(df, "Gender"), key="reg_gender")
        occupation = st.selectbox("Occupation", get_unique_values(df, "Occupation"), key="reg_occupation")
        daily_screen_time = st.number_input(
            "Daily Screen Time",
            float(df["Daily_Screen_Time"].min()),
            float(df["Daily_Screen_Time"].max()),
            float(df["Daily_Screen_Time"].median()),
            key="reg_daily_screen"
        )
        social_media_usage = st.number_input(
            "Social Media Usage",
            float(df["Social_Media_Usage"].min()),
            float(df["Social_Media_Usage"].max()),
            float(df["Social_Media_Usage"].median()),
            key="reg_social_media"
        )

    with col2:
        night_usage = st.selectbox("Night Usage", [0, 1], format_func=binary_label, key="reg_night")
        sleep_hours = st.number_input(
            "Sleep Hours",
            float(df["Sleep_Hours"].min()),
            float(df["Sleep_Hours"].max()),
            float(df["Sleep_Hours"].median()),
            key="reg_sleep"
        )
        work_study_hours = st.number_input(
            "Work/Study Hours",
            float(df["Work_Study_Hours"].min()),
            float(df["Work_Study_Hours"].max()),
            float(df["Work_Study_Hours"].median()),
            key="reg_work"
        )
        physical_activity = st.selectbox(
            "Physical Activity",
            get_unique_values(df, "Physical_Activity"),
            key="reg_physical"
        )
        social_interaction_score = st.number_input(
            "Social Interaction Score",
            int(df["Social_Interaction_Score"].min()),
            int(df["Social_Interaction_Score"].max()),
            int(df["Social_Interaction_Score"].median()),
            key="reg_social_interaction"
        )

    with col3:
        caffeine_intake = st.number_input(
            "Caffeine Intake",
            int(df["Caffeine_Intake"].min()),
            int(df["Caffeine_Intake"].max()),
            int(df["Caffeine_Intake"].median()),
            key="reg_caffeine"
        )
        smoking = st.selectbox("Smoking", [0, 1], format_func=binary_label, key="reg_smoking")
        alcohol = st.selectbox("Alcohol", [0, 1], format_func=binary_label, key="reg_alcohol")
        depression = st.selectbox("Depression", [0, 1], format_func=binary_label, key="reg_depression")
        anxiety = st.selectbox("Anxiety", [0, 1], format_func=binary_label, key="reg_anxiety")

    input_regression = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Daily_Screen_Time": daily_screen_time,
        "Social_Media_Usage": social_media_usage,
        "Night_Usage": night_usage,
        "Sleep_Hours": sleep_hours,
        "Work_Study_Hours": work_study_hours,
        "Physical_Activity": physical_activity,
        "Social_Interaction_Score": social_interaction_score,
        "Caffeine_Intake": caffeine_intake,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "Depression": depression,
        "Anxiety": anxiety
    }])

    st.write("Input row:")
    st.dataframe(input_regression, use_container_width=True)

    if st.button("Predict Stress Level"):
        prediction = selected_regression_model.predict(input_regression)[0]

        st.subheader("Prediction Result")
        st.metric("Predicted Stress Level", f"{prediction:.2f}")

        X_regression = df.drop(columns=["Person_ID", "Stress_Level", "Burnout"])

        show_regression_shap(
            selected_regression_model,
            X_regression,
            input_regression
        )

    with st.expander("Show learning curve for selected model"):
        X_regression = df.drop(columns=["Person_ID", "Stress_Level", "Burnout"])
        y_regression = df["Stress_Level"]

        if st.button("Compute regression learning curve"):
            show_regression_learning_curve(
                selected_regression_model,
                X_regression,
                y_regression
            )