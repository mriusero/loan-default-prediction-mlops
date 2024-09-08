import os

import pandas as pd
import streamlit as st


def load_csv_files(parent_folder_path):
    with st.spinner('Loading data...'):
        # Crée un dictionnaire pour stocker les DataFrames
        dataframes = {}

        # Liste des sous-dossiers à parcourir
        subfolders = ['raw', 'processed']

        # Compte le nombre total de fichiers à traiter pour la barre de progression
        total_files = 0
        for subfolder in subfolders:
            subfolder_path = os.path.join(parent_folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for _, _, files in os.walk(subfolder_path):
                    for file in files:
                        if file.endswith('.csv'):
                            total_files += 1

        # Initialisation de la barre de progression
        progress_bar = st.progress(0)

        # Traitement des fichiers
        files_processed = 0
        for subfolder in subfolders:
            subfolder_path = os.path.join(parent_folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for root, _, files in os.walk(subfolder_path):
                    for file in files:
                        if file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            # Crée une clé basée sur le nom du fichier sans le chemin
                            df_name = os.path.splitext(file)[0]
                            dataframes[df_name] = pd.read_csv(file_path)

                            # Mise à jour de la barre de progression
                            files_processed += 1
                            progress = files_processed / total_files
                            progress_bar.progress(progress)

    return dataframes
