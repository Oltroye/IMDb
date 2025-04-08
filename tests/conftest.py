import pytest
import pandas as pd

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "review": ["This movie was great!", "I hated this movie.", "It was okay."],
        "sentiment": ["positive", "negative", "neutral"]
    })

@pytest.fixture
def sample_X():
    return pd.Series(["This movie was great!", "I hated this movie."])

@pytest.fixture
def sample_y():
    return pd.Series([1, 0])