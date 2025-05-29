"# Jewelry-Price-Optimization-with-ML-Project" 
Project Goal:
Build and evaluate machine learning models to predict jewelry prices, uncover key price drivers, and enable data-driven pricing for better business decisions at Gemineye.

🚀 Project Highlights
Data Cleaning & Feature Engineering:
Handled missing data, outliers, and transformed features (log-prices, date/time, categorical encodings).

Exploratory Data Analysis (EDA):
Visualized price distribution, detected skewness/kurtosis, explored category/gem/metal patterns, and identified correlations.

Modeling & Evaluation:

Linear Regression

Random Forest

XGBoost
Compared models using RMSE, MAE, and R².

Experiment Tracking:
Used MLflow to track all experiment runs, metrics, and pipeline artifacts for full reproducibility.

Business Insights:
Uncovered key drivers of price (gem type, metal, category, customer repeat status) to guide smarter pricing and inventory decisions.

🛠️ Tech Stack & Skills
Python, pandas, numpy

scikit-learn, XGBoost

matplotlib, seaborn

MLflow (for experiment tracking)

Jupyter Notebooks, Streamlit (optional app)

📊 Key Results
Best Models: Random Forest and XGBoost achieved R² > 0.95 and low RMSE on test data.

Business Impact:

More accurate pricing = higher revenue, reduced under/over-pricing risk

Ability to target high-value segments (by gem/metal/category)

Reproducible, auditable ML process

🔄 Getting Started
# 1. Clone the repository
git clone https://github.com/p-alaita/Jewelry-Price-Optimization-with-ML.git
cd Jewelry_Price_Optimization

# 2. (Optional) Set up a new conda environment
conda create -n jewelry_env python=3.9
conda activate jewelry_env

# 3. Install dependencies
pip install -r requirements.txt
# Or install manually:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost mlflow streamlit joblib

# 4. Run the analysis notebook
jupyter notebook notebooks/JP_Optimization.ipynb

# 5. Start the MLflow tracking UI (optional)
mlflow ui

# 6. (Optional) Run the Streamlit app
streamlit run app.py


📢 Business Recommendations
Focus on high-value gems and metals: Premium segments drive price.

Implement dynamic pricing: Use model predictions to optimize margins.

Regularly retrain models: Ensure pricing stays relevant as trends shift.


🤝 Acknowledgments
Project by Patience Alaita-Jerome
For Gemineye Pricing Team