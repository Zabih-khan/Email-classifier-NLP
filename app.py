import streamlit as st
import pickle 
from nltk.tokenize import word_tokenize

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

tfidfd = pickle.load(open('tfidf.pkl','rb'))

def classify_email(email_text):
    input_features = tfidfd.transform([email_text])
    prediction = model.predict(input_features)[0]
    return prediction

def main():
    st.title('Email Classifier 💌')

    # Add a text input for the user to enter email text
    email_text = st.text_area('Enter Email Text:', '')

    if st.button('Classify'):
        if email_text.strip() == '':
            st.error('Please enter some text.')
        else:
            prediction = classify_email(email_text)
            if prediction == 0:
                st.success('This email is likely not spam.✅')
            else:
                st.error('This email is likely spam. 🚩')

if __name__ == '__main__':
    main()
