from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        """
        Initialisation de la classe Model.
        """
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        """
        Entraîner le modèle sur les données d'entraînement.
        """
        X_train_transformed = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_transformed, y_train)

    def predict(self, X_test):
        """
        Prédire les résultats pour les données de test.
        """
        X_test_transformed = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_transformed)

    def evaluate(self, y_test, y_pred):
        """
        Évaluer les performances du modèle.
        """
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Précision du modèle : {accuracy:.2%}")
        print("\n📊 Rapport de classification :")
        print(classification_report(y_test, y_pred))

        # Visualisation des prédictions
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y_pred, palette=['green', 'red'])
        plt.title("Répartition des prédictions du modèle")
        plt.xlabel("Sentiment prédit")
        plt.ylabel("Nombre d'avis")
        plt.show()