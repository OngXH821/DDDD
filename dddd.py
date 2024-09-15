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

# Load the dataset (adjust the path to your dataset)
df = pd.read_csv('Dataset-SA.csv')

# Preprocessing: Remove stopwords and punctuation from the 'Review' column
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['Review'] = df['Review'].apply(preprocess_text)

X = df['Review']
y = df['Sentiment']

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title('Sentiment Analysis on Product Reviews')

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write("Classification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
st.text(classification_report(y_test, y_pred))

df_report = pd.DataFrame(report).transpose()
df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)

def plot_classification_report(report_df):
    categories = report_df.index.tolist()
    precision = report_df['precision'].tolist()
    recall = report_df['recall'].tolist()
    f1_score = report_df['f1-score'].tolist()

    x = range(len(categories))
    plt.figure(figsize=(10, 6))
    plt.bar(x, precision, width=0.2, label='Precision', align='center')
    plt.bar([p + 0.2 for p in x], recall, width=0.2, label='Recall', align='center')
    plt.bar([p + 0.4 for p in x], f1_score, width=0.2, label='F1-Score', align='center')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Classification Metrics')
    plt.xticks([p + 0.2 for p in x], categories)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

plot_classification_report(df_report)

def predict_sentiment(user_comment):
    processed_comment = preprocess_text(user_comment)
    user_comment_tfidf = tfidf.transform([processed_comment])
    prediction = model.predict(user_comment_tfidf)
    return prediction[0]

user_comment = st.text_input("Enter your product review:")

if user_comment:
    sentiment = predict_sentiment(user_comment)
    st.write(f"The sentiment of the comment is: {sentiment}")
