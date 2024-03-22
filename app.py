import streamlit as st
import string 
import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import unicodedata
import nltk
import plotly.express as px

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


model = joblib.load("logistic_regression.pkl")

def clean_text(sentence):
    text = sentence.lower()

    text = BeautifulSoup(text, 'html.parser').get_text()

    text = ' '.join([word for word in text.split() if not word.startswith('http')])

    text = ''.join([char for char in text if char not in string.punctuation + '’‘'])

    text = ''.join([i for i in text if not i.isdigit()])

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])

    return text

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    # prediction = model.predict([cleaned_text])
    if(model.predict([cleaned_text]) == 1):
        prediction = 'Positive'
    else:
        prediction = 'Negative'
    return prediction


def main():
    st.title("Sentiment Analysis of Product Reviews🔍")

    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])


    if uploaded_file is not None:
            df = None

            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error in Reading file: {e}")

            if df is not None:
                st.subheader("Uploaded Data")
                st.write(df.head())

                st.subheader("Sentiment Prediction")
                text_column = st.selectbox("Select the column for sentiment analysis", df.columns)

                predictions = df[text_column].apply(predict_sentiment)

                df['Predicted Sentiment'] = predictions
                st.write(df[[text_column, 'Predicted Sentiment']])

                st.subheader("Barchart for Reviews 📈")
                fig = px.bar(df,x='Predicted Sentiment',title='Predicted Sentiments Distribution')
                st.plotly_chart(fig, use_container_width=True)

                # st.subheader("Histogram")
                # hist_fig = px.histogram(df, x='Predicted Sentiment', nbins=30, title='Predicted Sentiments Distribution(Histogram)')
                # st.plotly_chart(hist_fig, use_container_width=True)

                st.subheader("Pie Chart ")
                pie_fig = px.pie(df, names='Predicted Sentiment', title='Predicted Sentiments Distribution (Pie Chart)')
                st.plotly_chart(pie_fig, use_container_width=True)

                positive_count = (df['Predicted Sentiment'] == 'Positive').sum()
                negative_count = (df['Predicted Sentiment'] == 'Negative').sum()

                if positive_count > negative_count:
                    st.title("Positive Reviews are More")
                    st.markdown('<div style="text-align: center;"><img src="https://emojicdn.elk.sh/🥳"></div>', unsafe_allow_html=True)

                elif positive_count < negative_count:
                    st.title("Negative Reviews are More")
                    st.markdown('<div style="text-align: center;"><img src="https://emojicdn.elk.sh/😓"></div>', unsafe_allow_html=True)

                else:
                    st.write("Both positive and negative sentiments have equal counts.")

    else:
        user_input = st.text_area("Enter a review text:")

        if st.button("Predict"):
            if user_input:
                sentiment = predict_sentiment(user_input)
                if sentiment == 'Positive':
                    st.markdown("<h2 style='text-align: center;'>Positive Sentiment Detected!</h2>", unsafe_allow_html=True)
                    st.markdown('<div style="text-align: center;"><img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnI3ODEyY3oyOTN3aWk0d2d2bzA0MXNvc251ejhkcmlpa3hlZ2hsNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/PPgZCwZPKrLcw75EG1/giphy.gif"></div>', unsafe_allow_html=True)
                elif sentiment == 'Negative':
                        st.markdown("<h2 style='text-align: center;'>Negative Sentiment Detected!</h2>", unsafe_allow_html=True)
                        st.markdown('<div style="text-align: center;"><img src="https://media2.giphy.com/media/ISOckXUybVfQ4/200.webp?cid=790b76110fa38wk1dy11b9dcy3wiw2nj1gdvh6jld7i6qb3g&ep=v1_gifs_search&rid=200.webp&ct=g"></div>', unsafe_allow_html=True)
                else:
                        st.error("Sentiment could not be determined.")
            else:
                st.warning("Please enter a review text.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Sentiment Analyzer App",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    main()


