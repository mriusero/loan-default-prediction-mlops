import streamlit as st


def page_2():
    st.markdown('<div class="header">#2 Feature Engineering_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is the feature engineering phase.")
    st.markdown('---')


    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## #Raw Data_")
        st.dataframe(st.session_state.data.dataframes['Loan_Data'])
    with col2:
        st.markdown("## #Statistics_")
        st.dataframe(st.session_state.data.dataframes['Loan_Data'].describe())

    st.markdown('---')
    st.markdown("## #Preprocessed Data_")
    col1, col2, col3 = st.columns(3)

    def display_data_info(data_key, col):
        with col:
            st.markdown(f"### #{data_key}_")
            preprocessed_data = st.session_state.processed_data
            df = preprocessed_data[data_key]
            n_rows, n_cols = df.shape
            st.write(df)
            st.write(f"Nombre de lignes : {n_rows}")
            st.write(f"Nombre de colonnes : {n_cols}")

            nan_percentage = df.isna().mean() * 100
            st.markdown("### Pourcentage de NaN par colonne")
            for column, percentage in nan_percentage.items():
                st.text(f"{percentage:.2f} : {column} %")

    # Utilisation de la fonction pour chaque colonne
    display_data_info('Loan_Data_train', col1)
    display_data_info('Loan_Data_val', col2)
    display_data_info('Loan_Data_test', col3)

