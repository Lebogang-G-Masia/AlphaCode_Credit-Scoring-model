# Credit Scoring Model

> This projects implements a credit scoring model for the financial customer behavior dataset. The goal is to classify a credit score as good or bad using finanicial and demographic data.

## Features

- Automatic dataset preprocessing (missing values, categorical encoding, feature scaling)
- Multiple classifiers trained and compared
- Performance evaluation using Precision, Recall, F1-Score, ROC-AUC
- Best model selection with cross validation
- Feature importance visualization in the streamlit app.
- User interface to input applicant details and get predictions in real-time.

## Installation

1. Clone the repo.

```bash
git clone https://github.com/Lebogang-G-Masia/AlphaCode_Credit-Scoring-model
cd AlphaCode_Credit-Scoring-model
```

2. Create virtual enviornment
```bash
python -m venv .venv
source .venv/bin/activate # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Training the model

#### Option 1: Auto-download the dataset
`python train_model.py --auto_download`

#### Option 2: Use local dataset
`python train_model.py --data_csv data/financial_customer_behaviour.csv --target y`

## Outputs
- Trained model -> `models/credit_model.pkl`
- Metadata -> `models/metadata.json`
- Console logs showing eval metrics

## Running the app
After training, run:
```bash
streamlit run app.py
```