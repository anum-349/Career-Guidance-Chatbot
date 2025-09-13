import streamlit as st
import joblib
import pandas as pd
from utils import preprocess, lemmatization
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load saved ML model and vectorizer
model = joblib.load('intent_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # ✅ Load trained vectorizer

# Load dataset
df = pd.read_csv(r'D:\Machine Learning\Project\Career_dataset.csv')
df.drop_duplicates(inplace=True)

# Preprocess dataset questions using same methods as during training
df['clean_question'] = df['question'].apply(lemmatization).apply(preprocess)

# Vectorize dataset questions using trained vectorizer
question_vectors = vectorizer.transform(df['clean_question'])

# Combine all answers as context for fallback generative QA
context = " ".join(df['answer'].astype(str).tolist())

# Initialize transformers QA pipeline (for fallback answers)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Streamlit UI
st.set_page_config(page_title="Career Guidance Chatbot", page_icon="🤖")
st.title("Artiflect Career Guidance Chatbot")
st.write("Ask me any career-related question, and I’ll guide you!")

# User input
user_input = st.text_input("Enter your question:")

if user_input:
    # Preprocess user input
    lemmatized = lemmatization(user_input)
    processed = preprocess(lemmatized)

    # Vectorize input using the same vectorizer used in training
    user_vector = vectorizer.transform([processed])

    # Predict career role
    predicted_role = model.predict(user_vector)[0]

    # Find most similar question in dataset
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_idx = similarities.argmax()
    best_answer = df.iloc[best_match_idx]['answer']
    best_match_question = df.iloc[best_match_idx]['question']
    best_match_role = df.iloc[best_match_idx]['role']
    match_score = similarities[0][best_match_idx]

    # Similarity threshold to decide if fallback answer is needed
    SIMILARITY_THRESHOLD = 0.3

    if match_score < SIMILARITY_THRESHOLD:
        st.info("⚠️ I couldn't find a close match in my dataset, generating an answer for you...")
        try:
            result = qa_pipeline(question=user_input, context=context)
            generated_answer = result['answer']
            st.markdown("### Generated Answer:")
            st.write(generated_answer)
        except Exception as e:
            st.error("Sorry, I couldn't generate an answer at this time.")
    else:
        st.markdown("### Best-Matched Answer:")
        st.write(best_answer)
        # Optional: Display matched question and role
        # st.markdown(f"**Matched Question:** _{best_match_question}_")
        # st.markdown(f"**Matched Role:** `{best_match_role}` | **Similarity Score:** `{match_score:.2f}`")

    st.markdown("---")
    st.markdown(f"### Suggested Career: **{predicted_role}**")