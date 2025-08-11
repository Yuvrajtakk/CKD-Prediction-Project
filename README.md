# Early Prediction for Chronic Kidney Disease (CKD)

This project is a machine learning application built to fulfill the requirements of the SmartInternz guided project. It follows the provided 7-milestone structure to develop a web application that can predict the early onset of Chronic Kidney Disease from patient data.

**Team:** Yuvraj Takk, Nirmal Choudhary

---

## Project Overview

Chronic Kidney Disease (CKD) is a major medical problem that can be cured if treated in the early stages. This project investigates various medical test attributes to build a predictive model that can help doctors detect CKD earlier, enabling timely treatment and preventing disease progression. The final product is a web application built with the Flask framework where a user can input patient data and receive a prediction from our trained Logistic Regression model.

---

## Project Structure

The repository is organized according to the guided project's template:

-   **/Training/**: Contains the dataset (`chronickidneydisease.csv`) and the Jupyter Notebook (`CKD_Model_Training.ipynb`) with the complete data cleaning, model training, and evaluation pipeline.
-   **/Flask/**: Contains the web application files.
    -   `app.py`: The main Flask server script that handles requests and predictions.
    -   `templates/`: Contains the HTML files for the user interface.
    -   `static/`: (Optional) For CSS and image files.
-   `CKD.pkl`: The final, saved Logistic Regression model, ready for use by the Flask app.
-   `.gitignore`: Specifies files for Git to ignore, such as the virtual environment.
-   `requirements.txt`: A list of all necessary Python libraries for this project.

---

## How to Run This Project

To run this project on your local machine, please follow these steps:

**1. Clone the Repository**
```bash
git clone [https://github.com/Yuvrajtakk/CKD-Flask-Project.git](https://github.com/Yuvrajtakk/CKD-Flask-Project.git)
cd CKD-Flask-Project