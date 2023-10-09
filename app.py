from flask import Flask, render_template, request
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data_fake = pd.read_csv('C:\\Users\\RISHI\\Documents\\VS Code\\Fake.csv')
data_true = pd.read_csv('C:\\Users\\RISHI\\Documents\\VS Code\\True.csv')


app = Flask(__name__)

# Load your pre-trained models and vectorization here
# You should load LR and DT models, and the TfidfVectorizer

# Load the models and vectorizer
LR = LogisticRegression()
DT = DecisionTreeClassifier()
vectorization = TfidfVectorizer()

# Load your pre-trained models and vectorization here
# You should load LR and DT models, and the TfidfVectorizer

# Load the models and vectorizer
LR = LogisticRegression()
DT = DecisionTreeClassifier()
vectorization = TfidfVectorizer()

# Function to preprocess text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Home page route
@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        news = request.form['t1']
        news = wordopt(news)
        new_x_test = [news]
        new_xv_test = vectorization.transform(new_x_test)
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        result = f"LR Prediction: {output_lable(pred_LR[0])}, DT Prediction: {output_lable(pred_DT[0])}"
    return render_template('index.html', result=result)

# Function to return the label
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

if __name__ == '__main__':
    app.run(debug=True)