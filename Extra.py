import random

random.seed(14872673)

random_integer = random.randint(1, 100)
print(random_integer)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud

# Ensure you have the required NLTK data files
nltk.download('vader_lexicon')

# Load the dataset
file_path = '/Users/default/Desktop/spotify52kData.csv'
data = pd.read_csv(file_path)

# Remove rows with missing values in 'album_name' and 'popularity'
data_cleaned = data.dropna(subset=['album_name', 'popularity'])

# Preprocess album titles
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove punctuation
    return text

data_cleaned['album_name_cleaned'] = data_cleaned['album_name'].apply(preprocess_text)

# Common Words in Album Titles
vectorizer = CountVectorizer(stop_words='english')
title_matrix = vectorizer.fit_transform(data_cleaned['album_name_cleaned'])
title_words = vectorizer.get_feature_names_out()
title_word_counts = title_matrix.sum(axis=0).A1
word_freq = pd.DataFrame({'word': title_words, 'count': title_word_counts})
word_freq = word_freq.sort_values(by='count', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='word', data=word_freq, palette='viridis')
plt.title('Most Common Words in Album Titles')
plt.xlabel('Count')
plt.ylabel('Word')
plt.show()

# Word Cloud of Album Titles
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data_cleaned['album_name_cleaned']))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Album Titles')
plt.show()

# Distribution of Album Title Lengths
data_cleaned['title_length'] = data_cleaned['album_name_cleaned'].apply(lambda x: len(x.split()))
plt.figure(figsize=(12, 6))
sns.histplot(data_cleaned['title_length'], bins=30, kde=True)
plt.title('Distribution of Album Title Lengths')
plt.xlabel('Title Length (number of words)')
plt.ylabel('Frequency')
plt.show()

# Sentiment Analysis of Album Titles
sia = SentimentIntensityAnalyzer()
data_cleaned['title_sentiment'] = data_cleaned['album_name_cleaned'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Categorize sentiment
data_cleaned['sentiment_category'] = pd.cut(data_cleaned['title_sentiment'], bins=[-1, -0.05, 0.05, 1], labels=['Negative', 'Neutral', 'Positive'])

# Popularity Analysis by Sentiment Categories
plt.figure(figsize=(12, 6))
sns.boxplot(x='sentiment_category', y='popularity', data=data_cleaned, palette='viridis')
plt.title('Popularity by Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Popularity')
plt.show()

# Average popularity by sentiment category
sentiment_popularity = data_cleaned.groupby('sentiment_category')['popularity'].mean()
print(sentiment_popularity)
