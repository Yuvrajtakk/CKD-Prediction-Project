# Early Prediction Model for Chronic Kidney Disease (CKD)

This project was developed by Yuvraj Takk and Nirmal Choudhary as part of the SmartInternz program, completed on an accelerated one-day timeline.

## The Problem
Chronic Kidney Disease (CKD) is a serious condition that often goes undiagnosed in its early stages. By the time symptoms appear, significant kidney damage may have already occurred. Our goal was to use machine learning to build a tool that could help doctors identify at-risk patients much earlier, based on standard patient health records.

## Our Approach
We tackled this project in three core phases:
1.  **Data Cleaning:** We started with a raw, real-world dataset and wrote a Python script to handle missing values, creating a reliable foundation for our model.
2.  **Modeling & Evaluation:** We trained a Logistic Regression model and used techniques like data scaling and SMOTE to specifically improve its ability to detect the minority class (patients with CKD).
3.  **Analysis & Reporting:** We analyzed the data to find key trends and summarized our entire process and findings in a final report.

## How to Run This Project
To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Yuvrajtakk/CKD-Prediction-Project.git](https://github.com/Yuvrajtakk/CKD-Prediction-Project.git)
    cd CKD-Prediction-Project
    ```

2.  **Set up the environment:**
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # Install the required libraries
    pip install pandas scikit-learn matplotlib seaborn imbalanced-learn joblib
    ```

3.  **Run the pipeline:**
    *Place the `kidney_disease.csv` file inside the `data/` folder.*
    ```bash
    # This first script cleans the data and saves the output
    python src/01_data_cleaning.py

    # This second script trains the model and saves the final version
    python src/02_model_training.py
    ```
4.  **View the results:**
    * The final trained model is saved in the `models/` folder.
    * The final report can be viewed in the `Final_CKD_Report.ipynb` notebook.

## Team
* **Yuvraj Takk**
* **Nirmal Choudhary**