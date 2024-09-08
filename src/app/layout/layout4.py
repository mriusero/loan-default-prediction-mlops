import streamlit as st
import json
import joblib as jb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
from ..components import get_mlruns_data, check_mlruns_directory, save_data_to_json

def page_4():
    st.markdown('<div class="header">#4 Prediction_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is the prediction phase, based on production models selected from Mlflow experiments.")
    st.markdown('---')

    if st.button('Predict Loan Default'):
        from src.app import get_data_splits
        X_train, y_train, X_val, y_val, X_test, y_test = get_data_splits()

        model_paths = {
            'Random Forest': 'models/random_forest_02.pkl',
            'XGBoost': 'models/xgboost_model_02.pkl',
            'LightGBM': 'models/lightgbm_02.pkl'
        }

        col1, col2, col3 = st.columns(3)

        for i, (model_name, path) in enumerate(model_paths.items()):
            model = jb.load(path)
            predictions = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive

            if i == 0:
                col = col1
            elif i == 1:
                col = col2
            else:
                col = col3

            with col:
                st.write(f'## {model_name}_:')

                # Calcul des métriques
                report_dict = classification_report(y_test, predictions, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.write(f'Classification Report pour {model_name}:')
                st.dataframe(report_df)

                # Matrice de confusion
                conf_matrix = confusion_matrix(y_test, predictions)
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Prédictions')
                ax.set_ylabel('Vérités')
                ax.set_title(f'Matrice de Confusion pour {model_name}')
                st.pyplot(fig)

                # Courbe ROC
                if len(set(y_test)) == 2:  # pour les cas de classification binaire
                    y_test_bin = label_binarize(y_test, classes=[0, 1])
                    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'Courbe ROC pour {model_name}')
                    ax.legend(loc='lower right')
                    st.pyplot(fig)

                # Courbe d'apprentissage
                train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy')
                train_mean = np.mean(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                fig, ax = plt.subplots()
                ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
                ax.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
                ax.set_xlabel('Nombre d\'échantillons d\'entraînement')
                ax.set_ylabel('Score')
                ax.set_title(f'Courbe d\'apprentissage pour {model_name}')
                ax.legend(loc='best')
                st.pyplot(fig)

                # Courbe de précision-rappel
                if len(set(y_test)) == 2:  # pour les cas de classification binaire
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
                    fig, ax = plt.subplots()
                    ax.plot(recall, precision, color='red', lw=2, label='Precision-Recall curve')
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.set_title(f'Courbe de Précision-Rappel pour {model_name}')
                    ax.legend(loc='best')
                    st.pyplot(fig)
    else:
        st.text("Click to generate reports & plots.")
