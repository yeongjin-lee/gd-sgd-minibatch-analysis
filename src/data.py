def make_train_matrix(
    df,
    test_size=0.2,
    random_state=42,
):
    """
    Follows the original notebook logic:
    - split X and y
    - split into train and test
    - apply preprocessing (scaling + one-hot encoding)
    - add bias term
    """
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

    # NOTE:
    # We fit the preprocessor only on the training set.
    # This matches the standard ML practice and avoids data leakage.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Add bias term (x0 = 1)
    X_train_b = np.c_[np.ones((X_train_processed.shape[0], 1)), X_train_processed]

    return X_train_b, y_train, (X_test, y_test), preprocessor
