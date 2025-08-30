
import json
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"

def load_dataset(csv_path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=NUMERIC_IMPUTE_STRATEGY)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=CATEGORICAL_IMPUTE_STRATEGY)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features

def save_metadata(path: str, meta: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_metadata(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
