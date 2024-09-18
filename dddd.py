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

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing reviews")

# Predict sentiment with all models
def predict_sentiment_with_all_models(user_comment):
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])

    # Load models
    naive_bayes_model = joblib.load('naive_bayes_model.joblib')
    svm_model = joblib.load('support_vector_machine_model.joblib')
    logistic_regression_model = joblib.load('logistic_regression_model.joblib')

    # Predict with each model
    predictions = {
        'Naive Bayes': naive_bayes_model.predict(user_comment_tfidf)[0],
        'SVM': svm_model.predict(user_comment_tfidf)[0],
        'Logistic Regression': logistic_regression_model.predict(user_comment_tfidf)[0]
    }
    
    return predictions

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
        
        # Load models
        naive_bayes_model = joblib.load('naive_bayes_model.joblib')
        svm_model = joblib.load('support_vector_machine_model.joblib')
        logistic_regression_model = joblib.load('logistic_regression_model.joblib')

        # Predict using all models
        uploaded_df['Sentiment_Naive_Bayes'] = naive_bayes_model.predict(X_uploaded_tfidf)
        uploaded_df['Sentiment_SVM'] = svm_model.predict(X_uploaded_tfidf)
        uploaded_df['Sentiment_Logistic_Regression'] = logistic_regression_model.predict(X_uploaded_tfidf)
        
        # Show predictions from all models
        st.write("### Sentiment Predictions from All Models:")
        st.dataframe(uploaded_df[['Review', 'Sentiment_Naive_Bayes', 'Sentiment_SVM', 'Sentiment_Logistic_Regression']])

# User input for predicting sentiment
user_comment = st.text_input("Enter your product review:")

if user_comment:
    # Get predictions from all models
    sentiments = predict_sentiment_with_all_models(user_comment)
    
    # Display sentiments from all models
    st.write(f"### Sentiment Predictions for Your Review:")
    for model_name, sentiment in sentiments.items():
        color = 'green' if sentiment == 'positive' else 'red'
        st.markdown(f"<p style='color:{color}; font-size:20px;'>*{model_name}:* {sentiment}</p>", unsafe_allow_html=True)
