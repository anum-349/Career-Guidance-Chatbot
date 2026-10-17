import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from utils import preprocess, lemmatization
import joblib

#Load Data Set
df = pd.read_csv('./Career_dataset.csv')
df.drop_duplicates(subset='question', inplace=True)

#Apply preprocessing on question series
df['clean_question'] = df['question'].apply(lemmatization).apply(preprocess)
df['role'] = df['role'].astype(str)

#convert question into vectors
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["clean_question"])
y = df['role']

#split train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#apply model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#predict results
y_predict = model.predict(x_test)

#Display accuracy of model
print("Accuracy Score: ", accuracy_score(y_test, y_predict))
print("F1 Score: ", f1_score(y_test, y_predict, average="weighted"))

#train model on whole data
model.fit(x, y)

#Save vectorizer and model into files
joblib.dump(model, 'intent_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')