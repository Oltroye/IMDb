from src.preprocessor import Preprocessor
from src.model import Model
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Étape 1 : Prétraitement des données
    preprocessor = Preprocessor()
    df = preprocessor.fit_transform("data/IMDB Dataset.csv")

    # Séparation des données en ensembles d'entraînement et de test
    X = df['review_cleaned']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Étape 2 : Entraînement du modèle
    model = Model()
    model.train(X_train, y_train)

    # Étape 3 : Prédictions et évaluation
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred)