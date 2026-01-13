import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Feature columns (Fish Market dataset)
NUM_COLS = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
CAT_COLS = ['Species']


def _make_dummy(seed: int = 42) -> pd.DataFrame:
    """
    Generates a small dummy dataset for quick smoke tests when Fish.csv is not available.
    This is optional and should not be used for reporting results.
    """
    rng = np.random.RandomState(seed)
    X_dummy = rng.uniform(10, 60, (150, 5))
    y_dummy = X_dummy[:, 0] * 30 + rng.normal(0, 100, 150)
    y_dummy[y_dummy < 0] = 0

    df = pd.DataFrame(
        np.column_stack((y_dummy, X_dummy)),
        columns=['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    )
    df['Species'] = rng.choice(['A', 'B', 'C'], 150)
    return df


def load_fish_dataframe(filepath: str = "data/Fish.csv", allow_dummy: bool = False) -> pd.DataFrame:
    """
    Loads Fish.csv and returns a cleaned pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to Fish.csv (not included in the repository).
    allow_dummy : bool
        If True, generate a dummy dataset when the file is missing (for smoke tests only).
    """
    try:
        df = pd.read_csv(filepath)
        # Match the original notebook preprocessing
        if 'Weight' in df.columns:
            df = df[df['Weight'] > 0]
        return df
    except FileNotFoundError:
        if allow_dummy:
            return _make_dummy()
        raise FileNotFoundError(
            f"File not found at '{filepath}'. Download Fish.csv (see data/README.md) and place it under data/."
        )


def make_train_matrix(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Builds the training design matrix following the original notebook logic:
    - split target/features
    - train/test split
    - fit preprocessing on train only (prevents leakage)
    - add bias term

    Returns
    -------
    X_train_b : np.ndarray
        Preprocessed training matrix with bias column.
    y_train : np.ndarray
        Training targets (shape: [N, 1]).
    (X_test, y_test) : tuple
        Raw test split (not preprocessed here; kept for possible extensions).
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.
    """
    if 'Weight' not in df.columns:
        raise ValueError("Expected column 'Weight' in the dataset.")

    y = df['Weight'].values.reshape(-1, 1)
    X = df.drop(columns=['Weight'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
        ],
        remainder='passthrough'
    )

    # Fit on training data only to avoid leakage
    X_train_processed = preprocessor.fit_transform(X_train)

    # Add bias term (x0 = 1)
    X_train_b = np.c_[np.ones((X_train_processed.shape[0], 1)), X_train_processed]

    return X_train_b, y_train, (X_test, y_test), preprocessor
