from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from wordcloud import WordCloud
import io
import base64
from matplotlib import pyplot as plt

app = Flask(__name__)
CORS(app)

# Download NLTK stopwords
nltk.download('stopwords')

# Load the TF-IDF vectorizer and sentiment model
try:
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open("sentiment_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    vectorizer = None
    model = None

def generate_word_cloud(text, sentiment):
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the plot to free up memory
    # Encode the image to base64 for embedding in HTML
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/')
def show():
    return render_template('index.html')

def preprocessing(text):
    emojis_pattern = re.compile(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)")
    stop_words = set(stopwords.words('english'))
    text = re.sub('<[^>]*>', '', text)  # Remove HTML tags
    emojis = emojis_pattern.findall(text)
    text = re.sub(r'[\W+]', ' ', text.lower()) + " ".join(emojis).replace('-', " ")
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not vectorizer or not model:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500

    try:
        data = request.json
        if 'mdata' not in data:
            return jsonify({"error": "Missing 'mdata' in request"}), 400

        pro_text = preprocessing(data['mdata'])
        vec_matrix = vectorizer.transform([pro_text])
        prediction = model.predict(vec_matrix)
        confidence = model.predict_proba(vec_matrix).max()
        sentiment_label = "positive" if prediction[0] == 1 else 'negative'
        wordcloud_image = generate_word_cloud(pro_text, sentiment_label)

        return jsonify({
            "message": "OK",
            "sentiment": int(prediction[0]),
            "confidence": float(confidence),
            "wordcloud": wordcloud_image
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)