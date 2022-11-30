import pandas as pd
from numpy import dot
from numpy.linalg import norm
import streamlit as st
from dadmatools.embeddings import get_embedding
import nltk
nltk.download('punkt')
from stqdm import stqdm
stqdm.pandas()

from sentence_transformers import SentenceTransformer
model_name = 'msmarco-MiniLM-L-6-v3'
model = SentenceTransformer(model_name)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_data():
    df = pd.read_csv('all_articles_2020_2022_11_15.csv').dropna().reset_index(drop=True)

    sentences = df.copy()
    sentences['sents'] = sentences['text'].apply(nltk.sent_tokenize)
    sentences = sentences[['sents']]
    sentences = sentences.explode('sents')
    sentences = sentences.reset_index().rename(columns={'index':'org_index'})

    sentences['embedding'] = sentences.sents.progress_apply(model.encode)
    return sentences,df
sentences,df = get_data()

def cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def search(search_term, sentences, entries=5, context_size=5):
    search_embedding = model.encode(search_term)

    sentences['sim_score'] = sentences['embedding'].progress_apply(lambda x: cosine_sim(search_embedding, x))
    sim = sentences.sort_values('sim_score', ascending=False)[0:entries].reset_index().rename(columns={'index':'org_idx'})

    def create_context(org, context_size):
        context = sentences.iloc[org].sents
        for i in range(context_size):
            if (i < len(sim)) and (i > 0):
                context = sentences.iloc[org-i].sents + '\n' + context
                context = context + '\n' + sentences.iloc[org+i].sents
        return context
    sim['context'] = sim['org_idx'].apply(lambda x: create_context(x, context_size))

    return sim

st.markdown('# Police1 Semantic Search')
st.markdown(
    '<small>Assembled by Peter Nadel, Tufts University</small>'
    ,unsafe_allow_html=True
)
st.markdown('<hr>', unsafe_allow_html=True)

search_term = st.text_input('Search any word or phrase.', 'Police')
context_size = st.number_input('Choose context size (number of sentences before and after).', min_value=1, value=2)
entries = st.number_input('Choose number of excerpts.', min_value=1, value=5)
sim = search(search_term, sentences, entries=entries, context_size=context_size)

st.markdown(
    f'<h2>{search_term}</h2>'
    ,unsafe_allow_html=True
)

for i in range(entries):
    st.markdown(
        f'<small>Similarity Score: {round(sim.sim_score.to_list()[i], 3)}</small>'
        ,unsafe_allow_html=True
    )
    st.markdown(
        f'<p>{sim.context.to_list()[i]}</p>'
        ,unsafe_allow_html=True
    )
    st.markdown('<hr>', unsafe_allow_html=True)