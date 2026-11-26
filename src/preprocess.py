
from pathlib import Path
import pandas as pd


def load_dataset():
    """Load the SMS Spam Collection dataset."""

    file_path = Path(__file__).resolve().parent.parent / "data" / "SMSSpamCollection"

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    return df


def clean_dataset(df):
    """Remove invalid and duplicate records."""

    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df["label"].isin(["ham", "spam"])]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = load_dataset()
    data = clean_dataset(data)

    print("Dataset loaded successfully!")
    print("\nDataset shape:", data.shape)
    print("\nClass distribution:")
    print(data["label"].value_counts())
    print("\nFirst five records:")
    print(data.head())
