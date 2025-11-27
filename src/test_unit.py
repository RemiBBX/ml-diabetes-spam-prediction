import pandas as pd

from src import detect_binary_and_continuous


def test_detect_binary_and_continuous():
    """'
    Test the function detect_binary_and_continuous() to ensure it correctly identifies
    binary and continuous columns in a given DataFrame.
    """
    # --- Simple dataframe ---
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],  # continuous
            "height": [1.70, 1.80, 1.65, 1.75],  # continous
            "is_male": [0, 1, 1, 0],  # binary
            "is_smoker": [1, 0, 0, 1],  # binary
        }
    )
    expected_binary = ["is_male", "is_smoker"]
    expected_continuous = ["age", "height"]

    binary_cols, continuous_cols = detect_binary_and_continuous(df)

    # --- Verification ---
    assert set(binary_cols) == set(expected_binary)
    assert set(continuous_cols) == set(expected_continuous)
