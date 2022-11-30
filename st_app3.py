import pandas as pd
from numpy import dot
from numpy.linalg import norm
import pyarrow as pa
import streamlit as st
import spacy
nlp = spacy.load('en_core_web_md')

def cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

source = pa.memory_map('police1-vecs.arrow', 'r')
table_mmap = pa.ipc.RecordBatchFileReader(source).read_all().column('vector')
vecs = table_mmap.to_pandas()

search = st.text_input('Search', 'Police')
search_emb = nlp(search).vector

res = vecs.apply(lambda x: cosine_sim(x, search_emb)).sort_values(ascending=False)[0:5]
st.write(res)
# table_mmap2 = pa.ipc.RecordBatchFileReader(source).read_all().column('sents')
# sents = table_mmap2.to_pandas()

# st.write(sents.iloc[res.index])
