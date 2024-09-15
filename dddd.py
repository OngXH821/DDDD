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

# Determine the unique labels
unique_labels = sorted(set(y_test) | set(y_pred))
print("Unique labels:", unique_labels)

# Confusion Matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
plot_confusion_matrix(cm, unique_labels)

# Predict sentiment
def predict_sentiment(user_comment):
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])
    prediction = model.predict(user_comment_tfidf)
    return prediction[0]

user_comment = st.text_input("Enter your product review:")

if user_comment:
    sentiment = predict_sentiment(user_comment)
    st.write(f"The sentiment of the comment is: {sentiment}")
