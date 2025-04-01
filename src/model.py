from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def train_model(df: pd.DataFrame) -> None:
    """
    Entraîne un modèle de classification des sentiments.
    """
    # Préparation des données
    df['review_cleaned'] = df['review'].str.lower()
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(df['review_cleaned'])

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, df['sentiment'], test_size=0.2, random_state=42
    )

    # Entraînement
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Précision du modèle : {accuracy:.2f}")
    print("\n📊 Rapport de classification :")
    print(classification_report(y_test, y_pred))

    return model, vectorizer