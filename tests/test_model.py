from src.model import Model

def test_train(sample_X, sample_y):
    model = Model()
    try:
        model.train(sample_X, sample_y)
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")

def test_predict(sample_X, sample_y):
    model = Model()
    model.train(sample_X, sample_y)
    predictions = model.predict(sample_X)
    assert len(predictions) == len(sample_X)

def test_evaluate(sample_X, sample_y):
    model = Model()
    model.train(sample_X, sample_y)
    predictions = model.predict(sample_X)
    try:
        model.evaluate(sample_y, predictions)
    except Exception as e:
        pytest.fail(f"Evaluation failed with error: {e}")