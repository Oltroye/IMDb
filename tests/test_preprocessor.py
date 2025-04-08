import pytest
from src.preprocessor import Preprocessor

def test_classify_sentiment():
    assert Preprocessor.classify_sentiment(0.1) == 'positive'
    assert Preprocessor.classify_sentiment(-0.1) == 'negative'
    assert Preprocessor.classify_sentiment(0.0) == 'neutral'

def test_fit(sample_df):
    preprocessor = Preprocessor()
    preprocessor.fit(df=sample_df)
    assert preprocessor.df is not None
    assert "review" in preprocessor.df.columns

def test_transform(sample_df):
    preprocessor = Preprocessor()
    preprocessor.fit(df=sample_df)
    transformed_df = preprocessor.transform()
    assert "review_length" in transformed_df.columns
    assert "vader_score" in transformed_df.columns
    assert "review_cleaned" in transformed_df.columns

def test_fit_transform(sample_df):
    preprocessor = Preprocessor()
    transformed_df = preprocessor.fit_transform(df=sample_df)
    assert len(transformed_df) == len(sample_df)
    assert "review_cleaned" in transformed_df.columns