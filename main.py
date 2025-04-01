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