#!/bin/sh
if [ -d '/gordon_police1_sem_search_st']; then
    echo "Running app..."
    streamlit run st_app.py
else
    git clone https://github.com/pnadelofficial/gordon_police1_sem_search_st.git
    cd gordon_police1_sem_search_st
    pip install -r requirements.txt
    echo 'Running app...'
    streamlit run st_app.py
fi