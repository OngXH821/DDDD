import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Download stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Dataset-SA.csv')

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

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

# Get unique labels
unique_labels = sorted(set(y_test) | set(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
plot_confusion_matrix(cm, unique_labels)

# Debug: Show sample data for y_test and y_pred
st.write("Sample y_test values:")
st.write(y_test.head())
st.write("Sample y_pred values:")
st.write(pd.Series(y_pred).head())

# Create DataFrames for actual and predicted sentiment counts
# Verify creation and renaming of columns
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
