# 📉 Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer
is likely to churn, with a live interactive web app for real-time predictions.

**Live demo:** [Customer Churn Prediction Streamlit App](https://customer-churn-prediction-shivampanday.streamlit.app/)
**Video walkthrough:** [End-to-End Customer Churn Prediction](https://www.youtube.com/watch?v=UrsalURTS9g)

---

## 🧠 What this project does

- Trains and compares 3 models (Logistic Regression, Random Forest, XGBoost) on the
  [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Automatically selects the best model based on F1-score
- Saves the trained model, scaler, and encoders for reuse
- Serves live predictions through a Streamlit web app with a clean form UI

## 🏗️ Tech Stack

- **Python**, **Pandas**, **NumPy** — data handling
- **scikit-learn**, **XGBoost** — model training and evaluation
- **Streamlit** — web app / deployment
- **joblib** — model persistence

## 📁 Project Structure

```
churn-prediction-app/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← you add this (see setup)
├── models/                                     ← generated after training
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_names.json
│   └── metrics.json
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Setup

1. **Clone this repo and create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download the dataset:**
   Get `WA_Fn-UseC_-Telco-Customer-Churn.csv` from
   [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   and place it inside the `data/` folder.

3. **Train the model:**
   ```bash
   python train_model.py
   ```
   This prints accuracy/F1/ROC-AUC for all 3 models and saves the best one
   to `models/`.

4. **Run the app locally:**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`.

## 🚀 Deploying for free (Streamlit Community Cloud)

1. Push this whole folder to a **public** GitHub repo — including the trained
   `models/` folder (small enough to commit directly; no need for Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set main file to `app.py` → **Deploy**.
4. You'll get a public URL like `https://your-app-name.streamlit.app` —
   put this on your resume next to this project.

## 📊 Results

After training, check `models/metrics.json` for exact numbers on your run.
Typical results on this dataset: ~80% accuracy, ~0.60 F1-score, ~0.84 ROC-AUC
(churn datasets are naturally imbalanced, so F1 and AUC matter more than
raw accuracy here — worth mentioning in interviews).

## ✍️ Resume bullet (once deployed)

> Built and deployed an end-to-end customer churn prediction model
> (XGBoost, 84% ROC-AUC) as a live Streamlit web app, enabling real-time
> churn-risk predictions from customer input.

## 🔮 Possible extensions (if you want to go further)

- Add SHAP values to explain *why* a customer is predicted to churn
- Add a batch-prediction mode (upload a CSV of many customers at once)
- Track experiments with MLflow
- Add a `Dockerfile` for containerized deployment
