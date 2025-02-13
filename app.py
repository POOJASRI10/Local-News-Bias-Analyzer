from flask import Flask, render_template, request, jsonify
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# Sentiment analysis function
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

# Keyword clustering function
def get_keywords(text):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# Wordcloud generation function
def generate_wordcloud(keywords):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    img_b64 = base64.b64encode(img.read()).decode('utf-8')
    return img_b64

# Route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to analyze the text
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the text from the form
    text1 = request.form['text1']
    text2 = request.form['text2']

    # Sentiment analysis
    sentiment_text1 = get_sentiment(text1)
    sentiment_text2 = get_sentiment(text2)

    # Keyword clustering
    keywords_text1 = get_keywords(text1)
    keywords_text2 = get_keywords(text2)

    # Generate word clouds
    wordcloud_text1 = generate_wordcloud(keywords_text1)
    wordcloud_text2 = generate_wordcloud(keywords_text2)

    # Prepare the analysis results
    analysis_result = {
        'sentiment_text1': sentiment_text1,
        'sentiment_text2': sentiment_text2,
        'keywords_text1': keywords_text1[:10],  # Show top 10 keywords
        'keywords_text2': keywords_text2[:10],  # Show top 10 keywords
        'wordcloud_text1': wordcloud_text1,
        'wordcloud_text2': wordcloud_text2
    }

    return render_template('result.html', result=analysis_result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
