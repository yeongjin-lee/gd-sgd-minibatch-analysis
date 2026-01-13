import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_COLS = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
CAT_COLS = ['Species']

def load_fish_dataframe(filepath: str = "data/Fish.csv", allow_dummy: bool = False) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df[df["Weight"] > 0]  # match your original notebook
    return df

def make_train_matrix(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    y = df['Weight'].values.reshape(-1, 1)
    X = df.drop('Weight', axis=1)

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

    X_train_processed = preprocessor.fit_transform(X_train)
    X_train_b = np.c_[np.ones((X_train_processed.shape[0], 1)), X_train_processed]

    # return 4 items (stable contract)
    return X_train_b, y_train, (X_test, y_test), preprocessor
