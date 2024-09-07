import streamlit as st

from ..components import handle_models

def page_1():
    st.markdown('<div class="header">#1 Exploratory Data Analysis_</div>', unsafe_allow_html=True)

    handle_models()

