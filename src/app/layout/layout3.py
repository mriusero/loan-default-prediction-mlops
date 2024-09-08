import streamlit as st
from ..components import handle_models


def page_3():
    st.markdown('<div class="header">#3 Experiments_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is the development phase with experiments, runs and track models from Mlflow.")
    st.markdown('---')


    st.markdown("# #Start a run_")
    handle_models()


