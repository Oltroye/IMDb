import pandas as pd

class IMDBPreprocessor:
    def __init__(self):
        """Initialisation du préprocesseur"""
        self.total_reviews = 0
        self.total_positive_reviews = 0
        self.total_negative_reviews = 0

    def fit(self, df):
        """Méthode d'adaptation qui apprend des informations sur les données"""
        # Analyse les données, par exemple, calcul des critiques positives et négatives
        self.total_reviews = len(df)
        self.total_positive_reviews = len(df[df['sentiment'] == 'positive'])
        self.total_negative_reviews = len(df[df['sentiment'] == 'negative'])
        print(f"Total critiques : {self.total_reviews}")
        print(f"Critiques positives : {self.total_positive_reviews}")
        print(f"Critiques négatives : {self.total_negative_reviews}")

    def transform(self, df):
        """Méthode de transformation des données"""
        df['length'] = df['review'].apply(len)  # Ajouter une colonne pour la longueur des critiques
        df['review_lower'] = df['review'].str.lower()  # Convertir les critiques en minuscules
        return df  # Retourner les données transformées

    def fit_transform(self, df):
        """Combine les étapes de fit et transform"""
        self.fit(df)  # Apprend des données
        return self.transform(df)  # Applique les transformations
