# 🌦️SkyCast-Predictive Analysis of Meteorological Events using Random Forest 🌦️

## 📌 Overview
This project applies Random Forest model to predict meteorological events such as rainfall, storms, and temperature fluctuations.  
It demonstrates the complete pipeline of **data preprocessing → model training → evaluation → deployment via a Streamlit app.**

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **ML Model:** Random Forest Classifier  
- **Deployment:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure
```
SkyCast/
├── app.py                      # Streamlit web app
├── model.pkl                   # Trained Random Forest model
├── Predictive_Analysis.ipynb   # Jupyter notebook with full workflow
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 🚀 How to Run Locally

1️⃣ Clone the repository:
```bash
git clone https://github.com/lizathulya/SkyCast.git
cd SkyCast
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Train the model (optional, if you want to regenerate `model.pkl`):
```bash
jupyter notebook Predictive_Analysis.ipynb
```

4️⃣ Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📈 Results
Random Forest achieved **72% accuracy** on the test dataset.  

Visualizations generated include:
- Feature importance plots  
- Confusion matrix  
- Prediction outputs for new inputs  

---

## ✨ Key Learnings
- Applied supervised learning with Random Forest  
- Avoided underfitting/overfitting using cross-validation  
- Evaluated model performance with accuracy, precision, recall, F1-score  
- Saved & loaded ML models using pickle  
- Built an interactive ML-powered web application  
