# Import des modules à exposer
from .app import streamlit_app
from .data import data_loader
from .features import feature_engine
from .models import model
from .visualization import visualize

# Définition de __all__ pour contrôler ce qui est importé avec "from project_name import *"
__all__ = ["streamlit_app", "data_loader", "feature_engine", "model", "visualize"]
