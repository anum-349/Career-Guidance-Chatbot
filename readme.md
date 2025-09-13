# Artiflect Career Guidance Chatbot

This project is a Career Guidance Chatbot that understands user career-related questions and suggests relevant career roles using Machine Learning classification combined with NLP-based question-answering capabilities.

---

## Features

- Trains a Logistic Regression model to classify career roles from user questions.
- Uses TF-IDF vectorization for text feature extraction.
- Uses Spacy and NLTK for text preprocessing (lemmatization, stopword removal, punctuation cleanup).
- Retrieves the best matching answer from a dataset using cosine similarity.
- Falls back to a Transformer-based question-answering model (`deepset/roberta-base-squad2`) to generate answers for unseen questions.
- Streamlit frontend for interactive chatting.

---

## Project Structure

career_chatbot_project/
- app.py # Streamlit app code (chatbot interface)
- train_model.py # Script to train and save the ML model and vectorizer
- utils.py # Preprocessing functions (lemmatization, cleaning)
- intent_model.pkl # Saved Logistic Regression model (after training)
- vectorizer.pkl # Saved TF-IDF vectorizer (after training)
- Career_dataset.csv # Dataset with career questions, answers, and roles
- requirements.txt # Python dependencies
- README.md # This documentation

---

## Setup and Installation

1. **Clone the repository or download the code.**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download Spacy English model (if not already installed):**

```bash
python -m spacy download en_core_web_sm
```

4. **Run the training script to generate the model and vectorizer files:**

python train_model.py
Run the Streamlit app:

```bash
streamlit run app.py
```

5. **Usage**
`Enter any career-related question in the input box. The chatbot will predict a career role.It will show the best matching existing answer from the dataset. If no good match is found, it will generate an answer using a transformer-based QA model. Suggested career role is displayed.`

6. **Notes**
The first time you run the app with the transformer model, internet connection is required to download model weights.

Subsequent runs use cached models and do not require internet.

Adjust similarity threshold in app.py to tune fallback behavior.

7. **Contact**
For questions or support, contact: [anumkousar552@gmail.com](anumkousar552@gmail.com)

Also, remember to run:

```bash
python -m spacy download en_core_web_sm
```
to download required NLP resources.