
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"


def load_data():
    """Load the original telecom churn dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print(df.head())

    return df


def clean_data(df):
    """Clean TotalCharges and remove invalid rows."""
    df_clean = df.copy()

    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"],
        errors="coerce"
    )

    print("\nMissing values before cleaning:")
    print(df_clean.isnull().sum()[
        df_clean.isnull().sum() > 0
    ])

    df_clean = df_clean.dropna(
        subset=["TotalCharges"]
    ).copy()

    print("\nCleaned dataset shape:", df_clean.shape)
    print("Duplicate rows:", df_clean.duplicated().sum())

    return df_clean


def analyze_data(df):
    """Display basic churn statistics and visualizations."""
    print("\nChurn counts:")
    print(df["Churn"].value_counts())

    print("\nChurn percentage:")
    print(
        (df["Churn"].value_counts(normalize=True) * 100)
        .round(2)
    )

    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.show()

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    print("\nChurn percentage by contract:")
    print(contract_churn.round(2))

    contract_churn.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5)
    )
    plt.title("Customer Churn by Contract Type")
    plt.ylabel("Percentage of Customers")
    plt.tight_layout()
    plt.show()

    for column, title in [
        ("tenure", "Customer Tenure vs Churn"),
        ("MonthlyCharges", "Monthly Charges vs Churn"),
    ]:
        print(f"\n{column} analysis:")
        print(df.groupby("Churn")[column].agg(
            ["mean", "median", "count"]
        ).round(2))

        sns.boxplot(data=df, x="Churn", y=column)
        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    analyze_data(df_clean)
