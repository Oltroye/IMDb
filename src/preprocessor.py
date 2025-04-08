import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class Preprocessor:
    def __init__(self):
        """
        Initialisation de la classe Preprocessor.
        Télécharge les ressources nécessaires pour NLTK et initialise l'analyseur de sentiment.
        """
        nltk.download('vader_lexicon')
        nltk.download('punkt')
        self.sia = SentimentIntensityAnalyzer()
        self.df = None

    def fit(self, filepath=None, df=None):
     """
     Charger les données depuis un fichier CSV ou un DataFrame existant.
     """
     if df is not None:
        self.df = df
     elif filepath is not None:
        self.df = pd.read_csv(filepath)
     else:
            raise ValueError("Vous devez fournir soit un DataFrame (df), soit un chemin de fichier (filepath).")
    
     print("\n🔍 Valeurs manquantes par colonne :")
     print(self.df.isnull().sum())
     return self

    def transform(self):
        """
        Appliquer les transformations sur les données chargées.
        """
        if self.df is None:
            raise ValueError("Les données doivent être chargées avec `fit` avant d'utiliser `transform`.")
        
        # Ajouter des colonnes pour les transformations
        self.df['review_length'] = self.df['review'].apply(len)
        self.df['vader_score'] = self.df['review'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
        self.df['vader_sentiment'] = self.df['vader_score'].apply(self.classify_sentiment)
        self.df['review_cleaned'] = self.df['review'].str.lower()
        return self.df

    def fit_transform(self, filepath=None, df=None):
        """
        Charger les données et appliquer toutes les transformations en une seule étape.
        """
        self.fit(filepath=filepath, df=df)
        return self.transform()

    @staticmethod
    def classify_sentiment(score):
        """
        Classifier un score de sentiment en positif, négatif ou neutre.
        """
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'