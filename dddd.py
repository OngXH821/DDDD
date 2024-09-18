import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
import string
import joblib  # Import joblib for saving/loading models
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO

# Download stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Dataset-SA.csv')

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):  # Check if text is a string
        return ''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['Review'] = df['Review'].apply(preprocess_text)

# Prepare data for modeling
X = df['Review']
y = df['Sentiment']

# TF-IDF transformation
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Create models: Naive Bayes, SVM, and Logistic Regression
models = {
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500)
}

# Train each model and save them
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.joblib')

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

# Streamlit app header
st.title('Sentiment Analysis on Product Reviews')

# Display the total number of reviews before preprocessing
st.write(f"*Total Number of Reviews before Preprocessing:* {len(df)}")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing reviews")

# Predict sentiment with the selected model (Naive Bayes as default)
def predict_sentiment(user_comment, model):
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])
    prediction = model.predict(user_comment_tfidf)
    return prediction[0]

# Handle file upload
if uploaded_file is not None:
    # Read and preprocess the uploaded file
    uploaded_df = pd.read_csv(uploaded_file)
    if 'Review' not in uploaded_df.columns:
        st.error("The uploaded file must contain a 'Review' column.")
    else:
        # Ensure 'Review' column is a string
        uploaded_df['Review'] = uploaded_df['Review'].astype(str)
        uploaded_df['Review'] = uploaded_df['Review'].apply(preprocess_text)
        X_uploaded = uploaded_df['Review']
        X_uploaded_tfidf = tfidf.transform(X_uploaded)
        
        # Load the Naive Bayes model by default
        model = joblib.load('naive_bayes_model.joblib')
        y_pred_uploaded = model.predict(X_uploaded_tfidf)
        uploaded_df['Sentiment'] = y_pred_uploaded
        
        # Calculate sentiment distribution
        sentiment_distribution = uploaded_df['Sentiment'].value_counts()
        sentiment_labels = sentiment_distribution.index
        sentiment_sizes = sentiment_distribution.values

        # Define colors for sentiment categories
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightskyblue']

        # Calculate percentages
        sentiment_percentages = sentiment_sizes / sentiment_sizes.sum() * 100

        # Plot pie chart
        st.write("### Sentiment Distribution Pie Chart (Uploaded File):")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sentiment_percentages, labels=sentiment_labels, autopct='%1.1f%%', startangle=140, colors=colors[:len(sentiment_labels)])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        # Display the review count table
        review_count_table = pd.DataFrame({'Sentiment': sentiment_labels, 'Review Count': sentiment_sizes})
        st.write("### Review Count Table (Uploaded File):")
        st.table(review_count_table)

# User input for predicting sentiment
user_comment = st.text_input("Enter your product review:")

if user_comment:
    # Load the Naive Bayes model
    model = joblib.load('naive_bayes_model.joblib')
    sentiment = predict_sentiment(user_comment, model)
    
    # Define color based on sentiment
    color = 'green' if sentiment == 'positive' else 'red'
    
    # Display sentiment with color
    st.markdown(f"<p style='color:{color}; font-size:20px;'>*The sentiment of the comment is:* {sentiment}</p>", unsafe_allow_html=True)

# The pie chart is displayed only if a file is uploaded
