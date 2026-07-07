from pathlib import Path
import pandas as pd
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training and testing datasets.
    Return:
        A tuple containing the training and testing DataFrames.
    """
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(train_df.columns):
        raise ValueError("train.csv must contain 'text' and 'label' columns.")
    if not required_columns.issubset(test_df.columns):
        raise ValueError("test.csv must contain 'text' and 'label' columns.")
    return train_df, test_df