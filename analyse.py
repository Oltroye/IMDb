import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 📌 Télécharger les ressources NLTK si ce n'est pas déjà fait
nltk.download('vader_lexicon')
nltk.download('punkt')

# 📌 Charger le dataset IMDB
df = pd.read_csv("IMDB Dataset.csv")

# 📌 Vérifier les valeurs manquantes
print("\n🔍 Valeurs manquantes par colonne :")
print(df.isnull().sum())

# 📌 Analyser la répartition des sentiments
print("\n⚖️ Répartition des sentiments :")
print(df['sentiment'].value_counts())

# 📌 Ajouter une colonne de longueur des critiques
df['review_length'] = df['review'].apply(len)

# 📌 Afficher les statistiques de longueur des critiques
print("\n📏 Statistiques sur la longueur des critiques :")
print(df['review_length'].describe())

# 📊 **Visualisation des données**
plt.figure(figsize=(12, 5))

# 1️⃣ Histogramme des longueurs des critiques
plt.subplot(1, 2, 1)
sns.histplot(df['review_length'], bins=30, kde=True, color='blue')
plt.title("Distribution de la longueur des critiques")
plt.xlabel("Longueur (nombre de caractères)")
plt.ylabel("Nombre de critiques")

# 2️⃣ Graphique en barres des sentiments
plt.subplot(1, 2, 2)
sns.countplot(x=df['sentiment'], palette=['green', 'red'])
plt.title("Répartition des critiques positives et négatives")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de critiques")

plt.tight_layout()
plt.show()

# **Analyse de sentiment avec NLTK VADER**
sia = SentimentIntensityAnalyzer()

# Appliquer VADER sur chaque critique et récupérer le score compound
df['vader_score'] = df['review'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Déterminer la classification des sentiments avec VADER
def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_sentiment'] = df['vader_score'].apply(classify_sentiment)

# Compter les avis neutres, positifs et négatifs
sentiment_counts = df['vader_sentiment'].value_counts()
print("\n📊 Répartition des critiques selon VADER :")
print(sentiment_counts)

# **Visualisation des résultats de VADER**
plt.figure(figsize=(6, 4))
sns.countplot(x=df['vader_sentiment'], palette=['green', 'gray', 'red'])
plt.title("Répartition des critiques selon le sentiment VADER")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de critiques")
plt.show()

# **TF-IDF pour vectoriser les critiques**
df['review_cleaned'] = df['review'].str.lower()

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(df['review_cleaned'])

# **Séparer les données en train/test**
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, df['sentiment'], test_size=0.2, random_state=42
)

# **Entraîner un modèle de classification (régression logistique)**
model = LogisticRegression()
model.fit(X_train, y_train)

# **Prédictions sur les données test**
y_pred = model.predict(X_test)

# **Évaluation du modèle**
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Précision du modèle : {accuracy:.2%}")

# **Rapport détaillé des performances**
print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred))

# **Visualisation des prédictions**
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred, palette=['green', 'red'])
plt.title("Répartition des prédictions du modèle")
plt.xlabel("Sentiment prédit")
plt.ylabel("Nombre d'avis")
plt.show()
