import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.title("Topic Modeling of Social Issues")
st.write("This application uses LDA (Latent Dirichlet Allocation) to discover hidden topics in social issue texts.")

# Load dataset
df = pd.read_csv("social_issues.csv")

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(preprocess)

# Vectorization
vectorizer = CountVectorizer(max_df=0.95, min_df=2)
X = vectorizer.fit_transform(df["clean_text"])

# LDA Model
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

st.subheader("Top Topics Identified")

feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    st.markdown(f"### Topic {topic_idx + 1}")
    top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
    st.write(", ".join(top_words))

st.success("Topic Modeling Completed Successfully!")
