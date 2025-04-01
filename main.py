import pandas as pd
from src.preprocessor import IMDBPreprocessor
from src.model import IMDBModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger les données
df = pd.read_csv("data/IMDB Dataset.csv")

# Initialisation du préprocesseur et application du prétraitement
preprocessor = IMDBPreprocessor()
df_transformed = preprocessor.fit_transform(df)  # Apprend des données puis applique les transformations

# Séparer les données en features (X) et target (y)
X = df_transformed['review_lower']  # Les critiques en minuscules
y = df_transformed['sentiment']  # Labels (sentiment positif/négatif)

# Vectorisation des critiques textuelles en format numérique
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Séparer les données en ensemble d'entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Initialisation du modèle
model = IMDBModel()

# Entraîner le modèle
model.train(X_train, y_train)

# Évaluer le modèle
model.evaluate(X_test, y_test)
