# Loan Default Prediction

![Banner](outputs/DecisionTree_confusion_matrix.png)

**Predicting Loan Defaults using Decision Tree & XGBoost**

This project predicts whether a loan applicant will default based on historical financial and demographic data. It demonstrates an **end-to-end machine learning workflow**, from data preprocessing to model evaluation and visualizations.

---

## 📊 Dataset

The dataset contains the following features:

ID, year, loan_limit, Gender, approv_in_adv, loan_type, loan_purpose, Credit_Worthiness,
open_credit, business_or_commercial, loan_amount, rate_of_interest, Interest_rate_spread,
Upfront_charges, term, Neg_ammortization, interest_only, lump_sum_payment, property_value,
construction_type, occupancy_type, Secured_by, total_units, income, credit_type, Credit_Score,
co-applicant_credit_type, age, submission_of_application, LTV, Region, Security_Type, Status, dtir1

yaml
Copy code

- **Target Column:** `Status` (loan approved / default)

---

## 🗂 Project Structure

loan-default-prediction/
│
├─ main.py # Entry point: preprocessing, training, evaluation
├─ requirements.txt # Python dependencies
├─ data/
│ └─ loan_data.csv # Input dataset
├─ outputs/ # Generated plots, CSVs, and images
└─ src/
├─ train_models.py # Model training functions
└─ evaluate_models.py # Model evaluation and visualization

yaml
Copy code

---

## ⚙ How to Run

1. Clone the repository:

```bash
git clone https://github.com/irfana-aslam/loan-default-prediction.git
cd loan-default-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the project:

bash
Copy code
python main.py
All outputs (plots, CSVs, images) will be saved in the outputs/ folder.

📈 Evaluation & Visualizations
1. Confusion Matrix
Decision Tree:


XGBoost:


Confusion matrices show the true vs. predicted loan status.

2. ROC Curve
Decision Tree:


XGBoost:


ROC curves visualize the trade-off between true positive rate and false positive rate. Higher AUC = better model.

3. Feature Importance
Decision Tree Top 20 Features:


XGBoost Top 20 Features:


Feature importance plots highlight the most influential features in predicting loan defaults. CSV files are also saved for further analysis.

💡 Key Highlights
End-to-end ML workflow: preprocessing → training → evaluation → visualization

Handles categorical variables and sanitizes column names for XGBoost

Produces outputs for reporting: CSV + PNG plots

Portfolio-ready: demonstrates ML skills and data visualization

🛠 Tech Stack
Python 3.10

Pandas, NumPy

scikit-learn, XGBoost

Matplotlib, Seaborn

🚀 Next Steps / Extensions
Hyperparameter tuning for better performance

Deploy the model as a REST API using Flask or FastAPI

Interactive dashboards for model monitoring

Add more algorithms like Random Forest or LightGBM

👩‍💻 About Me
Hi! I am Irfana Aslam, a Python developer and machine learning enthusiast. I love building projects that combine data analysis, machine learning, and visualization. I create portfolio-ready projects that showcase real-world skills.

📫 Contact Me
Email: irfanaaslam69@gmail.com

Alternate Email: irfanaaslam0786@gmail.com

LinkedIn: linkedin.com/in/irfana-aslam

GitHub: github.com/irfana-aslam

🎯 References
XGBoost Documentation

scikit-learn Documentation

Pandas Documentation

📂 Outputs
All outputs (generated after running main.py) are saved in the outputs/ folder:

Copy code
outputs/
├─ DecisionTree_confusion_matrix.png
├─ DecisionTree_ROC_curve.png
├─ DecisionTree_feature_importance.png
├─ DecisionTree_feature_importance.csv
├─ XGBoost_confusion_matrix.png
├─ XGBoost_ROC_curve.png
├─ XGBoost_feature_importance.png
├─ XGBoost_feature_importance.csv
├─ project_banner.png