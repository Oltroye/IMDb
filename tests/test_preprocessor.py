import pytest
from src.preprocessor import classify_sentiment

def test_classify_sentiment():
    assert classify_sentiment(0.1) == 'positive'
    assert classify_sentiment(-0.1) == 'negative'
    assert classify_sentiment(0.0) == 'neutral'