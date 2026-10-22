
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_analysis import load_data, clean_data
from preprocessing import prepare_data, build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def train_model():
    # Load and clean data
    df = load_data()
    df_clean = clean_data(df)

    # Prepare train/test data
    X_train, X_test, y_train, y_test = prepare_data(df_clean)

    # Build complete model pipeline
    preprocessor = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    # Train model
    model.fit(X_train, y_train)

    # Save trained model
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "churn_model.joblib"
    joblib.dump(model, model_path)

    print("\nModel training completed successfully!")
    print("Saved model to:", model_path)

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()
