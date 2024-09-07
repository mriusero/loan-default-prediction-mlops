import os

import pandas as pd


class CSVLoader:
    def __init__(self, folder_path: str):
        """
        Initializes the loader with the path to the folder containing the CSV files.
        """
        self.folder_path = folder_path

    def load_csv_files(self):
        """
        Loads all CSV files from the specified folder and returns a dictionary
        of DataFrames where the keys are the filenames without the extension.
        """
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        dataframes = {}

        for file in csv_files:
            file_path = os.path.join(self.folder_path, file)
            df_name = os.path.splitext(file)[0]
            dataframes[df_name] = pd.read_csv(file_path)

        return dataframes
