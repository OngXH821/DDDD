# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import joblib  # Import joblib for saving/loading models
import streamlit as st
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')

# Streamlit app header
st.title('Sentiment Analysis with Model Comparison')

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing product reviews")

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):  # Check if text is a string
        return ''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Handle file upload and model evaluation
if uploaded_file is not None:
    # Read dataset and display basic info
    df = pd.read_csv(uploaded_file)
    st.write(f"Total Number of Reviews: {len(df)}")
    
    # Apply preprocessing
    df['Review'] = df['Review'].apply(preprocess_text)
    
    # Split data into features (X) and labels (y)
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
    
    # Dictionary to store accuracy scores
    model_accuracies = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy
        
        # Display classification report
        st.write(f"\n### {model_name} Model ###")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")
        st.text(classification_report(y_test, y_pred))
    
    # Display model accuracies in a table
    accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])
    st.write("### Model Accuracy Comparison")
    st.table(accuracy_df)
    
    # Plot accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    accuracy_df.set_index('Model').plot(kind='bar', color='skyblue', legend=False, ax=ax)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Save the trained models and the TF-IDF vectorizer for later use
    for name, model in models.items():
        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.joblib')
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

# User input for predicting sentiment
st.write("### Predict Sentiment for a Review:")
user_comment = st.text_input("Enter a product review:")

if user_comment:
    # Load the Naive Bayes model for prediction (or allow user to select a model)
    selected_model_name = st.selectbox("Choose a model for prediction", ['Naive Bayes', 'Support Vector Machine', 'Logistic Regression'])
    model = joblib.load(f'{selected_model_name.replace(" ", "_").lower()}_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    
    # Preprocess and predict sentiment
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])
    prediction = model.predict(user_comment_tfidf)
    
    # Display the predicted sentiment
    color = 'green' if prediction[0] == 'positive' else 'red'
    st.markdown(f"<p style='color:{color}; font-size:20px;'>*The sentiment of the comment is:* {prediction[0]}</p>", unsafe_allow_html=True)
