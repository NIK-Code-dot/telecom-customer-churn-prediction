
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def prepare_data(df_clean):
    """Split customer data into training and testing sets."""
    X = df_clean.drop(columns=["Churn", "customerID"])
    y = df_clean["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    """Build numerical and categorical preprocessing pipelines."""
    categorical_cols = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


if __name__ == "__main__":
    from data_analysis import load_data, clean_data

    df = load_data()
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data(df_clean)
    preprocessor = build_preprocessor(X_train)

    print("Preprocessing setup completed successfully!")
