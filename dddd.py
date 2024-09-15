import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import streamlit as st
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Dataset-SA.csv')

# Count the number of reviews in the dataset
total_reviews = len(df)

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['Review'] = df['Review'].apply(preprocess_text)

# Split data
X = df['Review']
y = df['Sentiment']

# TF-IDF transformation
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title('Sentiment Analysis on Product Reviews')

# Display the total number of reviews
st.write(f"Total Number of Reviews: {total_reviews}")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Debug: Show sample data for y_test and y_pred
st.write("Sample y_test values:")
st.write(y_test.head())
st.write("Sample y_pred values:")
st.write(pd.Series(y_pred).head())

# Create DataFrames for actual and predicted sentiment counts
actual_counts = pd.DataFrame(y_test.value_counts()).reset_index()
actual_counts.columns = ['Sentiment', 'Count_Actual']

predicted_counts = pd.DataFrame(pd.Series(y_pred).value_counts()).reset_index()
predicted_counts.columns = ['Sentiment', 'Count_Predicted']

# Display DataFrames for debugging
st.write("Actual counts DataFrame:")
st.write(actual_counts)

st.write("Predicted counts DataFrame:")
st.write(predicted_counts)

# Ensure the Sentiment columns are properly aligned and match
if 'Sentiment' in actual_counts.columns and 'Sentiment' in predicted_counts.columns:
    actual_counts['Sentiment'] = actual_counts['Sentiment'].astype(str)
    predicted_counts['Sentiment'] = predicted_counts['Sentiment'].astype(str)

    # Merge actual and predicted counts
    sentiment_comparison = pd.merge(actual_counts, predicted_counts, on='Sentiment', how='outer').fillna(0)

    # Plot actual vs predicted sentiment comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_comparison.plot(kind='bar', x='Sentiment', ax=ax, color=['skyblue', 'orange'])
    plt.title('Actual vs Predicted Sentiment Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.error("The 'Sentiment' column is missing in one of the DataFrames.")
    if 'Sentiment' not in actual_counts.columns:
        st.error("The 'Sentiment' column is missing in actual_counts DataFrame.")
    if 'Sentiment' not in predicted_counts.columns:
        st.error("The 'Sentiment' column is missing in predicted_counts DataFrame.")

# Predict sentiment
def predict_sentiment(user_comment):
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])
    prediction = model.predict(user_comment_tfidf)
    return prediction[0]

# User input for predicting sentiment
user_comment = st.text_input("Enter your product review:")

if user_comment:
    sentiment = predict_sentiment(user_comment)
    st.write(f"The sentiment of the comment is: {sentiment}")

# Calculate sentiment distribution
st.write("Sentiment Distribution:")
sentiment_distribution = df['Sentiment'].value_counts()
sentiment_labels = sentiment_distribution.index
sentiment_sizes = sentiment_distribution.values

# Calculate percentages
sentiment_percentages = sentiment_sizes / sentiment_sizes.sum() * 100

# Plot pie chart
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(sentiment_percentages, labels=sentiment_labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)
