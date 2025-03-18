import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


nltk.download('vader_lexicon')

df = pd.read_csv("IMDB Dataset.csv")

sia = SentimentIntensityAnalyzer()


df['vader_score'] = df['review'].apply(lambda x: sia.polarity_scores(x)['compound'])

def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_sentiment'] = df['vader_score'].apply(classify_sentiment)

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
