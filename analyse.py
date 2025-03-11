import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("IMDB Dataset.csv")

print("\nAperçu des 5 premières lignes :")
print(df.head())

print("\nDimensions du dataset :", df.shape)

df['length'] = df['review'].apply(len)

total = len(df)
positifs = len(df[df['sentiment'] == 'positive'])
pourcentage_positif = (positifs / total) * 100
print(f"\n✅ Il y a {positifs} commentaires positifs sur {total} ({pourcentage_positif:.2f}%)")

print("\nLongueur moyenne des critiques par sentiment :")
print(df.groupby('sentiment')['length'].mean())

plt.figure(figsize=(8, 5))
sns.boxplot(x='sentiment', y='length', data=df)
plt.title("Longueur des critiques selon le sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Longueur (nombre de caractères)")
plt.show()

df['review_lower'] = df['review'].str.lower()  

positive_words = ['great', 'amazing', 'love']
negative_words = ['boring', 'bad', 'terrible']

def count_word_presence(word, df):
    return df['review_lower'].str.contains(word).mean()

print("\nFréquence des mots positifs :")
for word in positive_words:
    print(f"- {word} : {count_word_presence(word, df):.2%}")

print("\nFréquence des mots négatifs :")
for word in negative_words:
    print(f"- {word} : {count_word_presence(word, df):.2%}")

df['contains_great'] = df['review_lower'].str.contains('great')
df['contains_bad'] = df['review_lower'].str.contains('bad')

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.barplot(x=df['sentiment'], y=df['contains_great'])
plt.title("Présence du mot 'great' par sentiment")

plt.subplot(1, 2, 2)
sns.barplot(x=df['sentiment'], y=df['contains_bad'])
plt.title("Présence du mot 'bad' par sentiment")

plt.tight_layout()
plt.show()
