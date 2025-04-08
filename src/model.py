from sklearn.feature_extraction.text import TfidfVectorizer
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
        # Vectoriser les données d'entraînement
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Entraîner le modèle
        self.model.fit(X_train_vectorized, y_train)

    def predict(self, X_test):
        """
        Effectuer des prédictions sur les données de test.
        """
        # Vectoriser les données de test
        X_test_vectorized = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vectorized)

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
        sns.countplot(x=y_pred, palette=['green', 'red'], hue=y_pred, dodge=False, legend=False)
        plt.title("Répartition des prédictions du modèle")
        plt.xlabel("Sentiment prédit")
        plt.ylabel("Nombre d'avis")
        plt.show()