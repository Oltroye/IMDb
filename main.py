from src.preprocessor import Preprocessor
from src.model import Model

if __name__ == "__main__":
    # Étape 1 : Prétraitement des données
    preprocessor = Preprocessor()
    df = preprocessor.fit_transform("data/IMDB Dataset.csv")

    # Étape 2 : Entraînement et évaluation du modèle
    model = Model()
    model.train(df)
    model.evaluate()

    train_test=train_test_split(data)
    x_train, y_train=preprocessor.fit_transform(train)
    model.train(x_train, y_train)
    
    x_test, y_test=preprocessor.fit_transform(test)
    y_pred=model.predict(x_test)
    model.evaluate(y_test, y_pred)