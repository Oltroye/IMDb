import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class IMDBModel:
    def __init__(self):
        """Initialisation du modèle"""
        self.model = LogisticRegression()  # Par défaut, on utilise une régression logistique
        self.vectorizer = None  # Un vectoriseur pour transformer les critiques textuelles

    def train(self, X, y):
        """Entraîne le modèle"""
        self.model.fit(X, y)  # Entraîner le modèle avec les données d'entrée X et les labels y
        print("Modèle entraîné.")

    def predict(self, X):
        """Prédit les sentiments sur de nouvelles critiques"""
        return self.model.predict(X)  # Prédit les sentiments (positifs/négatifs)

    def evaluate(self, X, y):
        """Évalue la performance du modèle"""
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)  # Calcule l'accuracy
        print(f"Accuracy du modèle : {accuracy * 100:.2f}%")
        return accuracy
