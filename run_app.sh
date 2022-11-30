#!/bin/sh
git clone https://github.com/pnadelofficial/gordon_police1_sem_search_st.git
cd gordon_police1_sem_search_st
pip install -r requirements.txt
streamlit run st_app.py