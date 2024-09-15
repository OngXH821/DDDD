import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import streamlit as st

# Download stopwords
nltk.download('stopwords')

# Load the dataset (adjust the path to your dataset)
df = pd.read_csv('Dataset-SA.csv')  # Ensure the dataset path is correct

# Preprocessing: Remove stopwords and punctuation from the 'Review' column
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply the preprocess function to the 'Review' column
df['Review'] = df['Review'].apply(preprocess_text)

# Split data into features (X) and labels (y)
X = df['Review']  # Using the 'Review' column as the feature
y = df['Sentiment']  # Using the 'Sentiment' column as the label

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app code
st.title('Sentiment Analysis on Product Reviews')

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display classification report
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Function to predict sentiment for new user input
def predict_sentiment(user_comment):
    # Preprocess the user comment
    processed_comment = preprocess_text(user_comment)
    # Transform it into TF-IDF features
    user_comment_tfidf = tfidf.transform([processed_comment])
    # Predict sentiment
    prediction = model.predict(user_comment_tfidf)
    return prediction[0]

# Get user input through Streamlit's text input
user_comment = st.text_input("Enter your product review:")

# Predict sentiment for the input comment when user enters text
if user_comment:
    sentiment = predict_sentiment(user_comment)
    st.write(f"The sentiment of the comment is: {sentiment}")
