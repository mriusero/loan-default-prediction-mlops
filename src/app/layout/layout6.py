import json
import numpy as np
import pickle
import random
import string
import streamlit as st

from ..components import get_mlruns_data, check_mlruns_directory, save_data_to_json

def page_6():
    st.markdown('<div class="header">#6 Production Model_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("This application predicts whether a customer will default on a loan based on their financial\nand personal information.")

    st.markdown('---')

    # Load the XGBoost model
    model_path = 'models/xgboost_model_06.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Calculate additional features
    def calculate_additional_features(loan_amt_outstanding, total_debt_outstanding, income, fico_score,
                                      fico_reference=700):

        debt_to_income_ratio = (total_debt_outstanding / income) * 100 if income > 0 else 0
        credit_to_income_ratio = (loan_amt_outstanding / income) * 100 if income > 0 else 0
        fico_score_diff = fico_score - fico_reference
        normalized_fico_score = (fico_score - 459) / (816 - 459)  # FICO score measured during experiments from 459 to 816 (to change in case of a new training)
        return debt_to_income_ratio, credit_to_income_ratio, fico_score_diff, normalized_fico_score

    # Description
    st.write("""### Please enter the customer details below""")
    credit_lines_outstanding = st.number_input("Number of outstanding credit lines", min_value=0, step=1)
    loan_amt_outstanding = st.number_input("Loan amount outstanding (€)", min_value=0.0, step=1000.0)
    total_debt_outstanding = st.number_input("Total debt outstanding (€)", min_value=0.0, step=1000.0)
    income = st.number_input("Annual income (€)", min_value=0.0, step=1000.0)
    years_employed = st.number_input("Years employed", min_value=0, step=1)
    fico_score = st.number_input("FICO score", min_value=300, max_value=850, step=1)

    # Calculate additional variables
    debt_to_income_ratio, credit_to_income_ratio, fico_score_diff, normalized_fico_score = calculate_additional_features(
        loan_amt_outstanding, total_debt_outstanding, income, fico_score
    )

    # Prepare the data for prediction
    features = np.array([[credit_lines_outstanding,
                          loan_amt_outstanding,
                          total_debt_outstanding,
                          income,
                          years_employed,
                          fico_score,
                          debt_to_income_ratio,
                          credit_to_income_ratio,
                          fico_score_diff,
                          normalized_fico_score]])

    if st.button("Predict Loan Default"):
        prediction = model.predict(features)

        # Display the result
        if prediction == 1:
            st.error("The customer is likely to default on the loan.")
        else:
            st.success("The customer is not likely to default on the loan.")