import streamlit as st
import json

from ..components import get_mlruns_data, check_mlruns_directory, save_data_to_json


def page_4():
    st.markdown('<div class="header">#4 Prediction_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is the prediction phase, based on production models selected from Mlflow experiments.")
    st.markdown('---')

