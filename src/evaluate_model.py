
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from data_analysis import load_data, clean_data
from preprocessing import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "churn_model.joblib"


def evaluate_model():
    # Check saved model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run src/train_model.py first."
        )

    # Prepare the same dataset and test split
    df = load_data()
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data(df_clean)

    model = joblib.load(MODEL_PATH)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nMODEL EVALUATION")
    print("=" * 35)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Threshold comparison
    print("\nTHRESHOLD COMPARISON")
    for threshold in [0.5, 0.4, 0.3]:
        y_threshold = (y_proba >= threshold).astype(int)
        print(f"\nThreshold: {threshold}")
        print("Precision:", round(
            precision_score(y_test, y_threshold, zero_division=0), 3
        ))
        print("Recall:", round(
            recall_score(y_test, y_threshold, zero_division=0), 3
        ))
        print("F1-score:", round(
            f1_score(y_test, y_threshold, zero_division=0), 3
        ))

    # Feature coefficient analysis
    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    coefficients = model.named_steps["classifier"].coef_[0]

    feature_analysis = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
    })

    print("\nTop features associated with higher churn:")
    print(feature_analysis.nlargest(10, "Coefficient").round(3))

    print("\nTop features associated with lower churn:")
    print(feature_analysis.nsmallest(10, "Coefficient").round(3))

    output_dir = PROJECT_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "feature_coefficients.csv"
    feature_analysis.to_csv(output_path, index=False)
    print("\nFeature analysis saved to:", output_path)


if __name__ == "__main__":
    evaluate_model()
