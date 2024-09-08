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
     * Success Metrics: F1 score, Accuracy, Precision, Recall.
     * Feasibility Assessment: given data availability and the ability to deploy on AWS, the project is feasible.
    """)

    st.markdown("""
    ### **3.2 Data Preparation**
    """)
    st.code("""
     * Data Cleaning: handling missing values and removing outliers.
     * Feature Engineering: created relevant features such as Debt-to-Income Ratio.
     * Data Splitting: 70% training, 15% validation, 15% testing to avoid overfitting and ensure generalizability.
    """)

    st.markdown("""
    ### **3.3 Model Engineering**
    """)
    st.code("""
     * Algorithms Used: Gradient Boosting Classifier, XGBoost Classifier.
     * Hyperparameter Tuning: used GridSearchCV to optimize model parameters.
     * Model Tracking with MLflow: each experiment was tracked in MLflow, capturing metrics such as AUC, F1 score, and model parameters.
    """)

    st.markdown("""
    ### **3.4 Model Evaluation**
    """)
    st.code("""
     * Evaluation Metrics: Accuracy, F1 Score, Precision, Recall.
     * Handling Overfitting: used cross-validation and regularization techniques.
     * Best Model: selected XGBoost Classifier with an F1 score of 0.78.
    """)

    st.markdown("""
    ### **3.5 Model Deployment**
    """)
    st.code("""
     * Deployment on AWS: model was containerized using Docker and deployed to an AWS EC2 instance.
     * Streamlit App: developed a user-friendly interface where users can input customer data to predict default risk.
     * CI/CD Pipeline: implemented using GitHub Actions to automate the deployment process.
    """)

    st.markdown("""
    ## **4. Monitoring and Maintenance**
    """)
    st.code("""
     * Model Monitoring: set up data drift detection and model performance monitoring.
     * Retraining Strategy: scheduled retraining with new data to maintain model accuracy.
     * Version Control: managed using Git to track changes in code, data, and models for reproducibility.
    """)

    st.markdown("""
    ## **5. Challenges and Solutions**
    """)
    st.code("""
     * Data Drift: implemented monitoring scripts to detect changes in data patterns over time.
     * Model Explainability: used SHAP values to understand model decisions, ensuring transparency.
     * Infrastructure Scaling: leveraged AWS auto-scaling features to handle increasing data volumes.
     * Bias Mitigation: regularly checked for model bias to ensure fairness and ethical outcomes.
     * Security Measures: employed AWS IAM roles and policies to secure models and data.
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
