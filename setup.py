from setuptools import setup, find_packages

setup(
    name="car-price-predictor",
    version="1.0.0",
    description="AI Used Car Price Predictor with 98.58% Accuracy",
    author="Syed Hussain Ali Shah",
    author_email="hussainrashdi456@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "streamlit>=1.10.0",
        "joblib>=1.1.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)