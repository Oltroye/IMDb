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

    def train(self, df):
        """
        Entraîner le modèle sur les données d'entraînement.
        """
        X = self.vectorizer.fit_transform(df['review_cleaned'])
        y = df['sentiment'].map({'positive': 1, 'negative': 0})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        self.X_test = X_test
        self.y_test = y_test

    def predict(self):
        """
        Effectuer des prédictions sur les données de test.
        """
        if not hasattr(self, 'X_test'):
            raise ValueError("Le modèle doit être entraîné avant de prédire.")
        return self.model.predict(self.X_test)

    def evaluate(self):
        """
        Évaluer les performances du modèle.
        """
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\n✅ Précision du modèle : {accuracy:.2%}")
        print("\n📊 Rapport de classification :")
        print(classification_report(self.y_test, y_pred))

        # Visualisation des prédictions
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y_pred, palette=['green', 'red'], hue=y_pred, dodge=False, legend=False)
        plt.title("Répartition des prédictions du modèle")
        plt.xlabel("Sentiment prédit")
        plt.ylabel("Nombre d'avis")
        plt.show()