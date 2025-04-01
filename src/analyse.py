import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def analyse_data(df: pd.DataFrame) -> None:
    """
    Analyse les avis IMDB et génère des visualisations.
    """
    # 1. Distribution des sentiments
    sentiment_distribution = df['sentiment'].value_counts()
    print("\n📊 Distribution des sentiments:")
    print(f"Positifs: {sentiment_distribution['positive']/len(df)*100:.2f}%")
    print(f"Négatifs: {sentiment_distribution['negative']/len(df)*100:.2f}%")

    # 2. Préparation pour l'analyse des mots
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def get_most_common_words(text_series, sentiment):
        # Combine tous les textes
        text = ' '.join(text_series)
        # Tokenisation
        words = word_tokenize(text.lower())
        # Suppression des stop words et ponctuation
        words = [word for word in words if word.isalnum() and word not in stop_words]
        # Comptage des mots
        word_freq = Counter(words).most_common(10)
        
        print(f"\n🔍 Mots les plus fréquents dans les avis {sentiment}:")
        for word, count in word_freq:
            print(f"{word}: {count}")
        
        return word_freq

    # 3. Analyse des mots les plus fréquents
    positive_reviews = df[df['sentiment'] == 'positive']['review']
    negative_reviews = df[df['sentiment'] == 'negative']['review']

    positive_words = get_most_common_words(positive_reviews, "positifs")
    negative_words = get_most_common_words(negative_reviews, "négatifs")

    # 4. Visualisations
    plt.figure(figsize=(12, 5))
    
    # Distribution des sentiments
    plt.subplot(1, 2, 1)
    sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values)
    plt.title("Distribution des Sentiments")
    plt.ylabel("Nombre d'avis")

    # Nuage de mots pour les avis positifs
    plt.subplot(1, 2, 2)
    text = ' '.join(positive_reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title("Nuage de mots - Avis Positifs")
    
    plt.tight_layout()
    plt.show()


df = pd.read_csv("data/IMDB_Dataset.csv")

df['review_cleaned'] = df['review'].str.lower()

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(df['review_cleaned'])

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, df['sentiment'], test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Précision du modèle : {accuracy:.2f}")


print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred, palette=['red', 'green'])
plt.title("Répartition des prédictions du modèle")
plt.xlabel("Sentiment prédit")
plt.ylabel("Nombre d'avis")
plt.show()

analyse_data(df)
