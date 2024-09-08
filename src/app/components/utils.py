import streamlit as st
from ...models import run_logr_pipeline, run_rf_pipeline

def get_data_splits():
    """
    Extrait les ensembles de données X_train, y_train, X_val, y_val, X_test, y_test
    à partir des données prétraitées.

    Args:
    - preprocessed_data (dict): Dictionnaire contenant les données prétraitées.
      Les clés doivent être 'Loan_Data_train', 'Loan_Data_val', 'Loan_Data_test'.

    Returns:
    - X_train (DataFrame): Données d'entraînement, sans la colonne cible.
    - y_train (Series): Valeurs cibles pour l'entraînement.
    - X_val (DataFrame): Données de validation, sans la colonne cible.
    - y_val (Series): Valeurs cibles pour la validation.
    - X_test (DataFrame): Données de test, sans la colonne cible.
    - y_test (Series): Valeurs cibles pour le test.
    """
    preprocessed_data = st.session_state.processed_data

    # Extraction des DataFrames prétraités
    df_train = preprocessed_data['Loan_Data_train']
    df_val = preprocessed_data['Loan_Data_val']
    df_test = preprocessed_data['Loan_Data_test']

    # Supposons que la colonne cible soit nommée 'target'
    target_column = 'target'

    # Séparation des caractéristiques et des cibles pour chaque ensemble de données
    X_train = df_train.drop(columns=['default'])
    y_train = df_train['default']

    X_val = df_val.drop(columns=['default'])
    y_val = df_val['default']

    X_test = df_test.drop(columns=['default'])
    y_test = df_test['default']

    return X_train, y_train, X_val, y_val, X_test, y_test


def handle_models():
    dataframes = st.session_state.processed_data
    loan_data_columns = dataframes['Loan_Data_train'].columns.tolist()
    loan_data_columns.remove('default')

    col1, col2, col3 = st.columns([4, 3, 2])

    with col1:
        model_name = st.selectbox("Choose a model", ["LogisticRegression", "RandomForestClassifier"])

    with col2:
        n_trials = st.number_input("Trials number for optimization if True", min_value=1, max_value=100, value=10)

    with col3:
        optimize = st.checkbox("Optimize hyperparameters", value=False)

    exp_name = f'exp_{model_name}'
    col1, col2, col3 = st.columns([4, 3, 2])
    with col1:
        X_selected = st.multiselect("Select X_", loan_data_columns, loan_data_columns[:])
        Y_selected = st.multiselect("Select Y_", 'default', 'default')
        run_button = st.button("Run prediction")
    with col2:
        st.markdown("##### X_")
        st.write(X_selected)

    with col3:
        st.markdown("##### Y_")
        st.write(Y_selected)

    if run_button:
        st.write(f"""
        * Running pipeline for {model_name}\n
            - Experience name: '{exp_name}'
            - Hyperparameters optimization: {'Yes' if optimize else 'No'}
            - Trials number: {n_trials}
        """)
        if model_name == 'LogisticRegression':
            run_logr_pipeline(optimize=optimize, n_trials=n_trials, exp_name=exp_name)

        if model_name == 'RandomForestClassifier':
            run_rf_pipeline(optimize=optimize, n_trials=n_trials, exp_name=exp_name)

