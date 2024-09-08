import streamlit as st


def page_0():
    st.markdown('<div class="header">#0 Project Overview_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is an overview of the project")
    st.markdown("---")
    st.markdown("""
    ## **1. Introduction**
    """)
    st.code("""
    Personal loans are a significant source of revenue for retail banks, but they come with inherent risks of defaults.

    Our goal is to build a predictive model to estimate the probability of default for each customer based on their 
    characteristics. Accurate predictions will help the bank allocate sufficient capital to cover potential losses, 
    thus ensuring financial stability.
    """)

    st.markdown("""
    ## **2. Project Overview**

    ### **Objective**
    """)
    st.code("""
     * Develop an end-to-end MLOps pipeline to predict loan defaults.
     * Deploy the best-performing model on Amazon Web Services (AWS) using Streamlit.
    """)

    st.markdown("""
    ### **Tools & Technologies**
    """)
    st.code("""
     * MLflow: model tracking and management.
     * Streamlit: interactive web app for model deployment.
     * AWS: cloud platform for hosting and scalability.
     * Git: version control for code and experiments.
     * Docker: containerization for consistent deployment.
    """)

    st.markdown("""
    ## **3. ML Lifecycle**

    ### **3.1 Planning**
    """)
    st.code("""
* Business Context: high default rates on personal loans threaten the bank's revenue.
 * Success Metrics:
   - *AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Measures the model's ability to distinguish between classes. A high AUC (close to 1) indicates good discrimination ability.
   - *Precision-Recall AUC (PR-AUC)**: Particularly useful for imbalanced data, this metric evaluates the precision and recall of the minority class. A high PR-AUC means the model maintains good precision while capturing most fraud cases.
   - *F1-Score**: A combined measure of precision and recall, useful for assessing the trade-off between false positives and false negatives. A good F1-score indicates a balance between these two types of errors.
   - *Recall**: A priority measure if the goal is to minimize false negatives (undetected frauds), as an undetected fraud represents a potential significant loss.
   - *Precision**: Important for minimizing false positives (legitimate transactions mistakenly classified as fraudulent), which could lead to unnecessary actions against good customers.
 * Feasibility Assessment: given data availability and the ability to deploy on AWS, the project is feasible.
    """)

    st.markdown("""
    ### **3.2 Data Preparation**
    """)
    st.code("""
     * Data Cleaning: handling missing values and removing outliers.
     * Feature Engineering: created relevant features such as Debt-to-Income Ratio.
     * Data Splitting: training, validation, testing split to avoid overfitting and ensure generalizability.
    """)

    st.markdown("""
    ### **3.3 Model Engineering**
    """)
    st.code("""
     * Algorithms Used: Random Forest Classifier, XGBoost Classifier, LightGBM Classifier.
     * Hyperparameter Tuning: used Optuna to optimize model parameters for all three classifiers.
     * Model Tracking with MLflow: each experiment was tracked in MLflow, capturing metrics such as Accuracy, F1 score, Precision, Recall, and model parameters.
    """)

    st.markdown("""
    ### **3.4 Model Evaluation**
    """)
    st.code("""
     * Evaluation Metrics: Accuracy, F1 Score, Precision, Recall.
     * Models Evaluated:
        - **Random Forest**: Accuracy = 0.9949, F1 Score = 0.9949, Precision = 0.9949, Recall = 0.9949.
        - **XGBoost**: Accuracy = 0.9964, F1 Score = 0.9964, Precision = 0.9964, Recall = 0.9964.
        - **LightGBM**: Accuracy = 0.9959, F1 Score = 0.9959, Precision = 0.9959, Recall = 0.9959.
     * Confusion Matrices:
        - **Random Forest**: True Positives = 339, True Negatives = 1604, False Positives = 5, False Negatives = 5.
        - **XGBoost**: True Positives = 341, True Negatives = 1605, False Positives = 4, False Negatives = 3.
        - **LightGBM**: True Positives = 341, True Negatives = 1604, False Positives = 5, False Negatives = 3.
     * Best Model: selected XGBoost Classifier with an F1 score of 0.9964.
    """)

    st.markdown("""
    ### **3.5 Model Deployment**
    """)
    st.code("""
     * Deployment on AWS: model was containerized using Docker and deployed to an AWS ECR instance.
     * Streamlit App: developed a user-friendly interface where users can input customer data to predict default risk.
     * CI/CD Pipeline: implemented using GitHub Actions to automate the deployment process.
    """)

    st.markdown("""
    ## **6. Benefits of MLOps Implementation**
    """)
    st.code("""
     * Improved Efficiency: automated model training, deployment, and monitoring.
     * Collaboration: enhanced collaboration between data scientists and DevOps teams.
     * Faster Time to Deployment: reduced model deployment time from weeks to hours.
     * Reduced Errors: consistent environments using Docker and CI/CD.
     * Governance and Compliance: maintained model versioning and audit trails for compliance.
    """)

    st.markdown("""
    ## **7. Conclusion**
    """)
    st.code("""
    By implementing an end-to-end MLOps pipeline, we successfully developed and deployed a robust model to predict loan defaults 
    in the retail banking sector. This project demonstrates the importance of MLOps in delivering reliable, scalable, and 
    explainable machine learning solutions that drive business value.

""")
