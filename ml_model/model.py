import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from collections import Counter
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
data = pd.read_csv('F:/Internship Project/Sentiment Analysis/Sentiment_A.csv')
data = data.iloc[:10000, :2]

# Visualize data distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
data['label'].value_counts().plot(kind='bar')
plt.title('Label Distribution (Bar)')

plt.subplot(1, 2, 2)
data['label'].value_counts().plot(kind='pie', labels=["positive", "negative"], colors=['g', 'r'], autopct='%0.1f%%', shadow=True, startangle=45, radius=1.3, explode=(0, 0.09))
plt.title('Label Distribution (Pie)')
plt.show()

# Check class distribution
print("Class Distribution:\n", data['label'].value_counts())

# Text preprocessing
emoji_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocessing(text):
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Extract emojis
    emojis = emoji_pattern.findall(text)
    # Remove special characters and convert to lowercase
    text = re.sub(r'[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', ' ')
    # Lemmatize and remove stopwords
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(text)

# Apply preprocessing to the text column
data['text'] = data['text'].apply(preprocessing)

# Positive and negative data
positivedata = data[data['label'] == 1]['text']
negativedata = data[data['label'] == 0]['text']

# Word frequency analysis
positive_words = ' '.join(positivedata).split()
positive_word_count = Counter(positive_words)
common_positive_words = positive_word_count.most_common(10)

negative_words = ' '.join(negativedata).split()
negative_word_count = Counter(negative_words)
common_negative_words = negative_word_count.most_common(10)

# Visualizing word frequency
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh([word for word, _ in common_positive_words], [count for _, count in common_positive_words], color='green')
axes[0].set_xlabel("Frequency")
axes[0].set_ylabel("Words")
axes[0].set_title("Positive Data Word Frequency")
axes[0].invert_yaxis()

axes[1].barh([word for word, _ in common_negative_words], [count for _, count in common_negative_words], color='red')
axes[1].set_xlabel("Frequency")
axes[1].set_ylabel("Words")
axes[1].set_title("Negative Data Word Frequency")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# TF-IDF Vectorization with n-grams
tfidf = TfidfVectorizer(strip_accents=None, lowercase=True, preprocessor=None, use_idf=True, norm='l2', smooth_idf=True, ngram_range=(1, 2))
y = data.label.values
x = tfidf.fit_transform(data['text'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2, shuffle=False)

# Logistic Regression with Cross-Validation and class weights
clf = LogisticRegressionCV(cv=6, scoring='accuracy', random_state=0, n_jobs=-1, verbose=3, max_iter=500, class_weight='balanced').fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Inspect feature importance
feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]
top_positive_words = [feature_names[i] for i in coefs.argsort()[-10:][::-1]]
top_negative_words = [feature_names[i] for i in coefs.argsort()[:10]]
print("Top Positive Words:", top_positive_words)
print("Top Negative Words:", top_negative_words)

# Save the trained model
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)