import spacy
import nltk
from nltk.corpus import stopwords
import string  

nltk.download('stopwords') # download stopwords
stop_words = set(stopwords.words('english')) #assign english stopwords only to stop_words
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    text = text.lower()  # lowercase
    text = ''.join([ch for ch in text if ch not in string.punctuation])  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(tokens)

#convert words to their base words
def lemmatization(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc])