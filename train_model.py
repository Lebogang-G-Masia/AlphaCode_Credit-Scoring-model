
import argparse
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from utils import load_dataset, build_preprocessor, save_metadata

def maybe_download_dataset(save_path: Path) -> Path:
    """
    Try to download the UCI 'Default of Credit Card Clients' dataset.
    If this fails (no internet), we create a small synthetic demo dataset.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        import io, requests, zipfile

        # UCI dataset (credit card default): often mirrored; try a common mirror
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        xls_path = save_path.with_suffix(".xls")
        with open(xls_path, "wb") as f:
            f.write(r.content)

        df = pd.read_excel(xls_path, header=1)
        # Create a friendlier subset / rename
        df = df.rename(columns={"default payment next month": "credit_bad"})
        # Keep a subset of columns for simplicity
        keep = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","BILL_AMT1","BILL_AMT2",
                "PAY_AMT1","PAY_AMT2","credit_bad"]
        df = df[keep]
        df.to_csv(save_path, index=False)
        return save_path
    except Exception as e:
        # Fallback: generate synthetic data similar to financial behavior
        np.random.seed(123)
        N = 800
        age = np.random.randint(21, 70, size=N)
        income = np.random.lognormal(mean=10.2, sigma=0.55, size=N)
        dti = np.clip(np.random.beta(2, 5, size=N) * 1.5, 0.01, 1.5)
        inquiries = np.random.poisson(1.5, size=N)
        late12 = np.random.poisson(0.6, size=N)
        paydelay = np.abs(np.random.normal(2, 5, size=N)).round(1)
        loan = np.random.lognormal(mean=9.4, sigma=0.6, size=N)
        emp = np.random.choice(["employed","self-employed","unemployed","student"], size=N, p=[0.66,0.18,0.11,0.05])
        home = np.random.choice(["rent","own","mortgage","other"], size=N, p=[0.36,0.25,0.34,0.05])
        purpose = np.random.choice(["debt_consolidation","home_improvement","auto","medical","vacation","education"], size=N)
        mort = np.random.choice([0,1], size=N, p=[0.56,0.44])

        risk = (
            0.015 * (70 - age) +
            0.002 * (loan / 1000) +
            2.0 * dti +
            0.15 * inquiries +
            0.2 * late12 +
            0.03 * paydelay +
            0.2 * (home == "rent").astype(int) +
            0.15 * (emp == "unemployed").astype(int) -
            0.000004 * income + 
            0.1 * (mort == 1).astype(int)
        )
        prob_bad = 1 / (1 + np.exp(- ( -2.0 + risk )))
        y = (np.random.rand(N) < prob_bad).astype(int)
        df = pd.DataFrame({
            "age": age,
            "income": income.round(2),
            "employment_status": emp,
            "debt_to_income_ratio": dti.round(3),
            "num_credit_inquiries": inquiries,
            "late_payments_12m": late12,
            "avg_payment_delay_days": paydelay,
            "loan_amount": loan.round(2),
            "loan_purpose": purpose,
            "home_ownership": home,
            "has_mortgage": mort,
            "credit_bad": y
        })
        df.to_csv(save_path, index=False)
        return save_path

def evaluate_and_log(model, X_test, y_test, label="model"):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        # fallback for models without predict_proba
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics["confusion_matrix"] = cm
    print(f"[{label}] Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}  ROC-AUC: {metrics['roc_auc']}")
    return metrics, y_proba

def main():
    parser = argparse.ArgumentParser(description="Train Credit Scoring Model")
    parser.add_argument("--data_csv", type=str, default=None, help="Path to CSV data")
    parser.add_argument("--target", type=str, default="credit_bad", help="Target column")
    parser.add_argument("--auto_download", action="store_true", help="Auto-download or synthesize dataset if not provided")
    parser.add_argument("--models_dir", type=str, default="models", help="Where to save models")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Where to save plots/outputs")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset
    if args.data_csv:
        csv_path = Path(args.data_csv)
        if not csv_path.exists():
            print(f"Provided data_csv '{csv_path}' not found.", file=sys.stderr)
            sys.exit(1)
    elif args.auto_download:
        csv_path = Path("data/uci_or_synth_credit.csv")
        csv_path = maybe_download_dataset(csv_path)
        print(f"Dataset ready at: {csv_path}")
    else:
        print("Please provide --data_csv or --auto_download", file=sys.stderr)
        sys.exit(1)

    X, y = load_dataset(str(csv_path), target=args.target)

    # Build preprocessing
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Candidate models
    candidates = []
    candidates.append(("logreg", LogisticRegression(max_iter=1000)))
    candidates.append(("dtree",  DecisionTreeClassifier(random_state=42)))
    candidates.append(("rf",     RandomForestClassifier(n_estimators=300, random_state=42)))
    if HAS_XGB:
        candidates.append(("xgb", XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            random_state=42, reg_lambda=1.0, eval_metric="logloss", n_jobs=-1
        )))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    # Cross-validate and fit
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}
    fitted_models = {}

    for name, est in candidates:
        from sklearn.pipeline import Pipeline
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", est)])
        roc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_summary[name] = {
            "roc_auc_mean": float(np.mean(roc_scores)),
            "roc_auc_std": float(np.std(roc_scores)),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores))
        }
        print(f"[CV] {name}: ROC-AUC {np.mean(roc_scores):.3f}±{np.std(roc_scores):.3f} | F1 {np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}")
        # Fit on train
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

    # Pick best by ROC-AUC mean
    best_name = max(cv_summary, key=lambda k: cv_summary[k]["roc_auc_mean"])
    best_model = fitted_models[best_name]
    print(f"Best model by CV ROC-AUC: {best_name}")

    # Evaluate on test
    test_metrics, y_proba = evaluate_and_log(best_model, X_test, y_test, label=f"{best_name}_test")

    # Save ROC curve
    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve - {best_name}")
        plt.savefig(outputs_dir / f"roc_{best_name}.png", bbox_inches="tight")
        plt.close()

    # Save confusion matrix plot
    from sklearn.metrics import ConfusionMatrixDisplay
    y_pred = best_model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize=None)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig(outputs_dir / f"cm_{best_name}.png", bbox_inches="tight")
    plt.close()

    # Attempt permutation importance on a sample (may be slow; keep small)
    try:
        if hasattr(best_model.named_steps["model"], "feature_importances_"):
            # Tree-based model with importances after OneHot -> we need feature names
            ohe = best_model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
            num_cols = best_model.named_steps["preprocess"].transformers_[0][2]
            cat_cols = best_model.named_steps["preprocess"].transformers_[1][2]
            cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names = list(num_cols) + cat_feature_names
            importances = best_model.named_steps["model"].feature_importances_
            importances = importances[:len(feature_names)]  # safety
            fi = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)[:30]
        else:
            # Permutation importance on a small subset
            X_small = X_test.sample(min(200, len(X_test)), random_state=42)
            y_small = y_test.loc[X_small.index]
            result = permutation_importance(best_model, X_small, y_small, n_repeats=5, random_state=42, n_jobs=-1)
            # Extract feature names after preprocessing
            ohe = best_model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
            num_cols = best_model.named_steps["preprocess"].transformers_[0][2]
            cat_cols = best_model.named_steps["preprocess"].transformers_[1][2]
            cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names = list(num_cols) + cat_feature_names
            importances = result.importances_mean[:len(feature_names)]
            fi = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)[:30]
        # Save top importances
        with open(outputs_dir / f"feature_importance_{best_name}.json", "w", encoding="utf-8") as f:
            json.dump([{"feature": k, "importance": float(v)} for k, v in fi], f, indent=2)
    except Exception as e:
        print(f"Could not compute feature importance: {e}")

    # Save model + metadata
    model_path = models_dir / "credit_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "best_model": best_name,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
        "target": args.target,
        "data_csv": str(csv_path),
    }
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model -> {model_path}")
    print(f"Saved metadata -> {models_dir / 'metadata.json'}")
    print(f"Saved outputs -> {outputs_dir}")

if __name__ == "__main__":
    main()
