import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self):
        pass

    def clean_data(self, dataframes):
        """
        Cleans data in DataFrames by removing outliers based on IQR factors and Z-scores.
        Prints removed rows for each column.
        """
        iqr_factors = {
            'loan_amt_outstanding': 2.80,
            'total_debt_outstanding': 3.90,
            'income': 1.90,
            'fico_score': 2.00
        }

        cleaned_dfs = {}
        for key, df in dataframes.items():
            df_cleaned = df.copy()

            for column, iqr_factor in iqr_factors.items():
                if column in df_cleaned.columns:
                    Q1 = df_cleaned[column].quantile(0.25)
                    Q3 = df_cleaned[column].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - iqr_factor * IQR
                    upper_bound = Q3 + iqr_factor * IQR

                    # Detect outliers using IQR
                    iqr_outliers = df_cleaned[column].lt(lower_bound) | df_cleaned[column].gt(upper_bound)
                    iqr_outlier_indices = df_cleaned.index[iqr_outliers]

                    # Detect outliers using Z-score
                    z_scores = stats.zscore(df_cleaned[column].fillna(0))
                    z_outliers = np.abs(z_scores) > 3
                    z_outlier_indices = df_cleaned.index[z_outliers]

                    # Combine outlier indices
                    all_outliers = np.unique(np.concatenate((iqr_outlier_indices, z_outlier_indices)))

                #    if len(all_outliers) > 0:
                #        print(f"Removing outliers for column '{column}' in DataFrame '{key}':")
                #        print(df_cleaned.loc[all_outliers, [column]])
                #        print("-" * 40)

                    df_cleaned = df_cleaned.loc[~df_cleaned.index.isin(all_outliers)]

            cleaned_dfs[key] = df_cleaned

        return cleaned_dfs

    def add_features(self, dataframes):
        """
        Adds features to the DataFrames.
        """
        enriched_dfs = {}
        for key, df in dataframes.items():
            # Calcul des nouvelles fonctionnalités
            df['debt_to_income_ratio'] = df['total_debt_outstanding'] / df['income']
            df['credit_to_income_ratio'] = df['loan_amt_outstanding'] / df['income']
            # df['credit_lines_per_year'] = df['credit_lines_outstanding'] / df['years_employed']
            df['fico_score_diff'] = df['fico_score'] - 700
            # df['debt_per_credit_line'] = df['total_debt_outstanding'] / df['credit_lines_outstanding']
            # df['employment_per_credit_line'] = df['years_employed'] / df['credit_lines_outstanding']
            df['normalized_fico_score'] = (df['fico_score'] - df['fico_score'].min()) / (
                    df['fico_score'].max() - df['fico_score'].min())

            print(f"Min fico score measured: '{df['fico_score'].min()}'")
            print(f"Max fico score measured: '{df['fico_score'].max()}'")

            # Conversion des colonnes entières en flottants
            int_columns = df.select_dtypes(include=['int64']).columns
            df[int_columns] = df[int_columns].astype(float)

            # Remplacer les infinies par NaN, puis supprimer les lignes avec des NaN
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            enriched_dfs[key] = df

        return enriched_dfs

    def split_data(self, dataframes, test_size=0.2, val_size=0.2, random_state=42):
        """
        Splits DataFrames into training, validation, and test sets.
        """
        split_dfs = {}

        for key, df in dataframes.items():
            X = df.drop('default', axis=1)
            y = df['default']

            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state)

            split_dfs[f'{key}_train'] = pd.concat([X_train, y_train], axis=1)
            split_dfs[f'{key}_val'] = pd.concat([X_val, y_val], axis=1)
            split_dfs[f'{key}_test'] = pd.concat([X_test, y_test], axis=1)

        return split_dfs

    @st.cache_data()
    def preprocess_data(_self):
        """
        Loads DataFrames from session_state, preprocesses them, and returns the results.
        """
        with st.spinner('Loading and preprocessing data...'):
            progress_bar = st.progress(0)

            dataframes = st.session_state.data.dataframes

            dataframes = _self.clean_data(dataframes)
            progress_bar.progress(33)

            dataframes = _self.add_features(dataframes)
            progress_bar.progress(66)

            dataframes = _self.split_data(dataframes)
            progress_bar.progress(100)

            return dataframes
