import streamlit as st
from ...models import run_logr_pipeline, run_rf_pipeline

def handle_models():
    st.title("Machine Learning pipeline")

    optimize = st.checkbox("Optimize hyperparameters", value=False)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        model_name = st.selectbox("Choose a model", ["LogisticRegression", "RandomForestClassifier"])

    with col2:
        n_trials = st.number_input("Trials number for study", min_value=1, max_value=100, value=10)

    exp_name = f'exp_{model_name}'

    if st.button("Run pipeline"):
        st.write(f"""
        * Running pipeline for {model_name}\n
            - Experience name: '{exp_name}'
            - Hyperparameters optimization: {'Yes' if optimize else 'No'}"
            - Trials number: {n_trials}
        """)

        if model_name == 'LogisticRegression':
            run_logr_pipeline(optimize=optimize, n_trials=n_trials, exp_name=exp_name)

        if model_name == 'RandomForestClassifier':
            run_rf_pipeline(optimize=optimize, n_trials=n_trials, exp_name=exp_name)


