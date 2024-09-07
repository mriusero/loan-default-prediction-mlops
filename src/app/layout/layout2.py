import streamlit as st

from ...data import CSVLoader

def page_2():
    st.markdown('<div class="header">#2 Empty page_</div>', unsafe_allow_html=True)

    loader = CSVLoader('./data/raw')
    dataframes = loader.load_csv_files()
    st.write(dataframes['Loan_Data'])