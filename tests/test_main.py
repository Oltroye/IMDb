import pytest
import pandas as pd
from src.preprocessor import add_review_length_column, calculate_positive_percentage
from src.model import count_word_presence

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'review': ['Great movie', 'Bad movie', 'Amazing film', 'Terrible film'],
        'sentiment': ['positive', 'negative', 'positive', 'negative']
    })

def test_add_review_length_column(sample_data):
    df = add_review_length_column(sample_data)
    assert 'length' in df.columns
    assert df['length'].iloc[0] == 12  # "Great movie" has 12 characters

def test_calculate_positive_percentage(sample_data):
    df = add_review_length_column(sample_data)
    positifs, total, pourcentage_positif = calculate_positive_percentage(df)
    assert positifs == 2
    assert total == 4
    assert 50.0 == pourcentage_positif

def test_count_word_presence(sample_data):
    df = sample_data.copy()
    df['review_lower'] = df['review'].str.lower()
    result = count_word_presence('great', df)
    assert result == 0.25  # "Great movie" appears 1 time out of 4 reviews
