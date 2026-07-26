# 🚗 AI Used Car Price Predictor

An end-to-end AI system that predicts fair market price of used cars with **98.58% accuracy** using XGBoost.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red)](https://your-app-link.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **R² Score** | **0.9858** (98.58%) |
| **MAE** | **$1,146** |
| **RMSE** | **$1,610** |
| **Prediction Time** | **0.3 seconds** |
| **Training Data** | **1M → 926k records** |

---

## 📁 Project Structure
AI_Car_Price_Predictor/
├── app/ # Streamlit web application
├── model/ # Trained models
├── data/ # Datasets
├── scripts/ # Utility scripts
├── notebooks/ # Jupyter notebooks
├── figures/ # Generated figures
├── outputs/ # Results
├── report/ # Project reports
├── tests/ # Unit tests
├── .github/ # GitHub Actions
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md


---

## 🚀 Quick Start

### Option 1: Python


# Clone repository
git clone https://github.com/yourusername/AI_Car_Price_Predictor.git
cd AI_Car_Price_Predictor

# Install dependencies
pip install -r requirements.txt

# Run app
cd app
streamlit run app.py
Option 2: Docker
bash
# Build image
docker build -t car-price-predictor ./app

# Run container
docker run -p 8501:8501 car-price-predictor
Option 3: Docker Compose
bash
docker-compose up --build
🔗 Links
Live Demo: https://your-app.streamlit.app

GitHub: https://github.com/yourusername/AI_Car_Price_Predictor

🛠️ Tech Stack
Category	Technology
ML Algorithm	XGBoost
Web Framework	Streamlit
Container	Docker
Data	Pandas, NumPy
Deploy	Streamlit Cloud, GitHub Actions
👥 Team
Syed Hussain Ali Shah

Muhammad Atif

Saqib Memon

🎓 Supervisor
Mr. Aashiq Shaikh
Shah Abdul Latif University, Khairpur (2026)