# 💳 Credit Card Approval Prediction System

An end-to-end **Machine Learning** web application that predicts whether a credit card application is **Approved** or **Rejected** based on an applicant's financial and personal information. The project covers the complete machine learning lifecycle, including data preprocessing, feature engineering, model training, evaluation, and deployment through a Flask web application.

---

# 🚀 Project Overview

Banks and financial institutions receive thousands of credit card applications every day. Manually evaluating each application is time-consuming, expensive, and prone to human error.

This project automates the credit card approval process using Machine Learning by analyzing applicant information and predicting whether the application is likely to be approved or rejected.

The application provides an easy-to-use web interface where users can enter applicant details and instantly receive a prediction.

---

# 🧠 Problem Statement

Financial institutions need a fast and reliable way to evaluate credit card applications while minimizing manual effort and maintaining consistency in decision-making.

This project aims to build an intelligent prediction system that:

* Automates credit card approval decisions.
* Reduces manual verification time.
* Improves decision consistency.
* Supports financial institutions with data-driven predictions.

---

# ✨ Features

* Credit Card Approval Prediction
* User-Friendly Web Interface
* Real-Time Prediction
* Machine Learning-Based Decision Making
* Secure Data Processing
* Responsive Bootstrap UI
* Dashboard Page
* About, FAQ, and Contact Pages
* Flask Web Deployment

---

# ⚙️ Technologies Used

* Python
* Flask
* Machine Learning (Scikit-learn)
* NumPy
* Pandas
* Joblib
* HTML5
* CSS3
* Bootstrap 5

---

# 🤖 Machine Learning Models

The following classification algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting (XGBoost)

The best-performing model was selected and saved for deployment using Joblib.

---

# 📊 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Label Encoding
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Model Selection
10. Model Saving
11. Flask Web Application Deployment

---

# 📁 Project Structure

```text
credit-card-approval/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── credit_card_model.pkl
│   ├── model.pkl
│   └── encoders.pkl
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── dashboard.html
│   ├── result.html
│   ├── about.html
│   ├── faq.html
│   └── contact.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── images/
│       ├── hero-banner.png
│       ├── favicon.png
│       ├── credit.png
│       └── download.png
│
├── application_record.csv
├── application_final.csv
├── credit_record.csv
├── credit_final.csv
├── Descriptive_analysis.py
├── Model_Building_code.py
├── removing_duplicates.py
└── train_model.ipynb
```

---

# 📥 Installation

### Clone the Repository

```bash
git clone https://github.com/VAKA-ART1/credit-card-approval.git

cd credit-card-approval
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

---

# 🌐 Web Application Pages

* Home
* Predict
* Dashboard
* About
* FAQ
* Contact

---

# 📈 Future Enhancements

* Credit Score Integration
* Fraud Detection Module
* Explainable AI (XAI)
* Real-Time Credit Bureau Integration
* Cloud Deployment
* User Authentication
* Admin Dashboard
* API Integration
* Model Performance Monitoring

---

# 🔚 Conclusion

The Credit Card Approval Prediction System demonstrates how Machine Learning can automate and improve the credit card application screening process. By analyzing financial and demographic information, the system provides fast, consistent, and reliable approval predictions that reduce manual effort and support better decision-making.

The project successfully integrates data preprocessing, feature engineering, machine learning model training, evaluation, and deployment into a complete end-to-end solution. The Flask-based web application offers an intuitive interface for real-time prediction, making the system practical for financial institutions as well as educational and research purposes.

Overall, this project showcases the effective use of Artificial Intelligence and Machine Learning to build an intelligent, scalable, and user-friendly credit card approval prediction system.
