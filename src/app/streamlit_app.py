import gc
import os

import streamlit as st

from .components import start_mlflow_ui

update_message = 'Data loaded'
display = ""

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main_layout():
    from .components import github_button
    from .layout import page_0, page_1, page_2, page_3, page_4

    st.set_page_config(
        page_title="SDA-MLOps",
        layout='wide',
        initial_sidebar_state="auto",
    )
    load_css()
    st.sidebar.markdown("# --- MLOps ---\n\n"
                        " ## *'Predicting Loan Default Risk Using MLOps in Retail Banking'*\n")
    page = st.sidebar.radio("Project_", ["#0 Introduction_",
                                         "#1 Exploratory Data Analysis_",
                                         "#2 Feature Engineering_",
                                         "#3 Experiments_",
                                         "#4 Prediction_",
                                         ])
    # -- LAYOUT --
    col1, col2 = st.columns([6, 4])
    with col1:
        global update_message
        st.markdown('<div class="title">MLOps</div>', unsafe_allow_html=True)
        st.markdown("#### *Predicting Loan Default Risk Using MLOps in Retail Banking* ")
        colA, colB, colC, colD = st.columns([2, 4, 4, 2])
        with colA:
            # st.text("")
            github_button('https://github.com/mriusero/projet-sda-mlops')
        with colB:
            # st.text("")
            st.text("")

            if st.button('Mlflow UI'):
                start_mlflow_ui()

        with colC:
            # st.text("")
            st.text("")
            st.link_button('Link 2',
                           'https://www.something.com')
        with colD:
            # st.text("")
            st.text("")
            #if st.button('Update data'):
            #    update_message = load_data(folder_path='')
            #    st.sidebar.success(f"{update_message}")
            #    print(update_message)
    with col2:
        st.text("")
        st.text("")
        st.text("")
        #ata = DataVisualizer()
        #t.session_state.data = data

    st.markdown('---')

    if page == "#0 Introduction_":
        page_0()
    elif page == "#1 Exploratory Data Analysis_":
        page_1()
    elif page == "#2 Feature Engineering_":
        page_2()
    elif page == "#3 Experiments_":
        page_3()
    elif page == "#4 Prediction_":
        page_4()

    st.sidebar.markdown("&nbsp;")

    gc.collect()
