# app.py
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Credit Scoring Model", layout="wide")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "credit_model.pkl"
META_PATH = MODELS_DIR / "metadata.json"

@st.cache_resource
def load_model_and_meta():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        st.error("Trained model or metadata not found. Please run `python train_model.py --auto_download` first.")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta

@st.cache_data
def load_dataframe_from_path(csv_path: str | None) -> pd.DataFrame | None:
    if not csv_path:
        return None
    p = Path(csv_path)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


PLACEHOLDER_SCHEMA = pd.DataFrame({
    "age":[30],
    "income":[50_000],
    "employment_status":["employed"],
    "debt_to_income_ratio":[0.35],
    "num_credit_inquiries":[1],
    "late_payments_12m":[0],
    "avg_payment_delay_days":[0.0],
    "loan_amount":[10_000],
    "loan_purpose":["debt_consolidation"],
    "home_ownership":["rent"],
    "marital_status":["single"],
    "education_level":["bachelor"],
    "region":["north"],
    "has_mortgage":[0],
})

NUMERIC_STEP_FRACTION = 100  # slider/step resolution

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def coerce_types_like_example(df: pd.DataFrame, example: pd.DataFrame) -> pd.DataFrame:
    """Coerce incoming single-row df to match dtypes of example_df where possible."""
    out = df.copy()
    for col in example.columns:
        if col not in out.columns:
            continue
        try:
            if pd.api.types.is_numeric_dtype(example[col]):
                out[col] = pd.to_numeric(out[col], errors="coerce")
            else:
                out[col] = out[col].astype("string")
        except Exception:
            pass
    return out

def _safe_min_max(s: pd.Series) -> tuple[float, float, float]:
    """Return (min, max, median) with safe fallbacks for degenerate columns."""
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if s_num.empty:
        return 0.0, 1.0, 0.5
    mn, mx, md = float(s_num.min()), float(s_num.max()), float(s_num.median())
    if np.isclose(mn, mx):
        # widen tiny ranges to make sliders usable
        pad = 1.0 if np.isclose(mn, 0.0) else abs(mn) * 0.1
        mn, mx = mn - pad, mx + pad
    return mn, mx, md

def build_input_ui(example_df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Applicant Details")

    # remove known target columns if present
    drop_cols = [c for c in example_df.columns if c.lower() in {"credit_bad", "target", "y"}]
    X = example_df.drop(columns=drop_cols, errors="ignore")

    user_vals: dict[str, object] = {}

    for col in X.columns:
        s = example_df[col]
        # Numeric input -> slider when feasible, else number_input
        if is_numeric_series(s):
            mn, mx, md = _safe_min_max(s)
            step = (mx - mn) / NUMERIC_STEP_FRACTION if mx > mn else 1.0
            # Use slider when range is reasonable
            if (mx - mn) <= 1e7:  # avoid huge sliders
                user_vals[col] = st.sidebar.slider(col, min_value=float(mn), max_value=float(mx), value=float(md), step=step)
            else:
                user_vals[col] = st.sidebar.number_input(col, value=float(md), min_value=float(mn), max_value=float(mx), step=step)
        else:
            # Categorical input -> selectbox
            choices = sorted(list(map(lambda x: "" if pd.isna(x) else str(x), s.dropna().unique().tolist())))
            if not choices:
                choices = [""]
            default_ix = 0
            user_vals[col] = st.sidebar.selectbox(col, options=choices, index=default_ix)

    return pd.DataFrame([user_vals])

def visualize_probability(prob_bad: float):
    """Simple probability bar using matplotlib (no extra deps)."""
    prob_bad = float(np.clip(prob_bad, 0.0, 1.0))
    fig, ax = plt.subplots(figsize=(6, 1.0))
    ax.barh([0], [prob_bad], align="center")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability of Bad Credit")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    st.pyplot(fig)

def get_feature_importances(model, meta, feature_names: list[str]) -> pd.DataFrame | None:
    # Priority 1: metadata provides mapping {feature: importance}
    meta_imp = None
    for key in ("feature_importances", "feature_importances_", "feature_importance"):
        if key in (meta or {}):
            meta_imp = meta[key]
            break

    if isinstance(meta_imp, dict) and meta_imp:
        imp = pd.Series(meta_imp, dtype=float)
        df = imp.reindex(feature_names).fillna(0.0).reset_index()
        df.columns = ["feature", "importance"]
        return df.sort_values("importance", ascending=False)

    # Priority 2: model has .feature_importances_ (tree-based)
    try:
        if hasattr(model, "feature_importances_"):
            arr = np.asarray(model.feature_importances_, dtype=float)
            if arr.size == len(feature_names):
                df = pd.DataFrame({"feature": feature_names, "importance": arr})
                return df.sort_values("importance", ascending=False)
    except Exception:
        pass

    # Priority 3: model has coef_ (linear/logistic)
    try:
        if hasattr(model, "coef_"):
            coef = np.ravel(model.coef_)
            if coef.size == len(feature_names):
                df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef), "signed_coef": coef})
                return df.sort_values("importance", ascending=False)
    except Exception:
        pass

    return None

def ensure_history_state():
    if "history_df" not in st.session_state:
        st.session_state.history_df = pd.DataFrame()

def append_to_history(applicant_row: pd.Series, pred_label: str, pred_int: int, prob_bad: float | None):
    ensure_history_state()
    row = applicant_row.to_dict()
    row.update({
        "_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_prediction": pred_label,
        "_pred_int": int(pred_int),
        "_prob_bad": float(prob_bad) if prob_bad is not None else np.nan,
    })
    st.session_state.history_df = pd.concat([st.session_state.history_df, pd.DataFrame([row])], ignore_index=True)

def download_button_for_history():
    ensure_history_state()
    if st.session_state.history_df.empty:
        st.info("No predictions yet.")
        return
    csv_bytes = st.session_state.history_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Prediction History (CSV)", data=csv_bytes, file_name="credit_scoring_history.csv", mime="text/csv")

def make_template_csv(example_df: pd.DataFrame) -> bytes:
    # Drop targets if present, and output a one-row template
    drop_cols = [c for c in example_df.columns if c.lower() in {"credit_bad", "target", "y"}]
    X = example_df.drop(columns=drop_cols, errors="ignore").head(1)
    return X.to_csv(index=False).encode("utf-8")


def main():
    st.title("Credit Scoring — Real-Time Prediction")
    st.write("Load a trained model and score new applicants with an interactive UI, visual probability, and insights.")

    model, meta = load_model_and_meta()
    st.sidebar.success(f"Loaded best model: **{meta.get('best_model','?')}**")

    st.subheader("Input Source")
    with st.expander("Choose schema source", expanded=True):
        example_file = st.file_uploader(
            "Optionally upload a CSV to infer the schema (one or more rows). If none, we'll fall back to the training CSV from metadata, else a placeholder.",
            type=["csv"]
        )

        if example_file is not None:
            example_df = pd.read_csv(example_file)
            source = "Uploaded CSV"
        else:
            data_csv = meta.get("data_csv")
            df_from_meta = load_dataframe_from_path(data_csv)
            if df_from_meta is not None:
                example_df = df_from_meta
                source = f"Metadata training CSV: {data_csv}"
            else:
                example_df = PLACEHOLDER_SCHEMA.copy()
                source = "Placeholder schema"

        st.caption(f"Schema source: **{source}**")
        st.write("Detected columns:", ", ".join(example_df.columns))

        st.download_button(
            "Download Input Template CSV",
            data=make_template_csv(example_df),
            file_name="applicant_template.csv",
            mime="text/csv",
            help="Download a one-row CSV matching the current schema to fill in offline."
        )

    tab_applicant, tab_predict, tab_insights, tab_history, tab_about = st.tabs(
        [" Applicant", " Prediction", " Insights", " History", "ℹ About"]
    )

    with tab_applicant:
        st.markdown("Applicant Preview")
        user_df = build_input_ui(example_df)
        user_df = coerce_types_like_example(user_df, example_df)  # gently align dtypes
        st.dataframe(user_df, use_container_width=True)

    with tab_predict:
        st.markdown("Make a Prediction")

        # Optional threshold (if we get probability, we can override classification)
        use_custom_threshold = st.toggle("Use custom decision threshold", value=False, help="Overrides the model's class using the predicted probability (if available).")
        threshold = 0.5
        if use_custom_threshold:
            threshold = st.slider("Bad-credit threshold (probability of bad ≥ threshold → classify as Bad)", 0.01, 0.99, 0.5, 0.01)

        # Predict
        if st.button("Predict Credit Risk", type="primary"):
            try:
                # Raw model prediction
                pred_int = int(model.predict(user_df)[0])
                prob_bad = None
                try:
                    # Assuming class 1 = bad, use [:, 1]
                    prob_bad = float(model.predict_proba(user_df)[0, 1])
                except Exception:
                    pass

                # Apply custom threshold if enabled and prob available
                if use_custom_threshold and (prob_bad is not None):
                    pred_int = int(prob_bad >= threshold)

                label = "Bad Credit (High Risk)" if pred_int == 1 else "Good Credit (Low Risk)"

                # Show result
                colA, colB = st.columns([1, 2])
                with colA:
                    st.metric("Prediction", label)
                    if prob_bad is not None:
                        st.metric("Probability of Bad", f"{prob_bad:.2%}")
                with colB:
                    if prob_bad is not None:
                        visualize_probability(prob_bad)
                    else:
                        st.info("Model does not provide probabilities (`predict_proba` unavailable).")

                # Save to history
                append_to_history(user_df.iloc[0], label, pred_int, prob_bad)
                st.success("Prediction added to history.")
            except Exception as e:
                st.exception(e)
                st.stop()

    with tab_insights:
        st.markdown("Model Performance (from metadata)")
        col1, col2 = st.columns(2)
        with col1:
            st.json(meta.get("test_metrics", {}))
        with col2:
            st.json(meta.get("cv_summary", {}))

        st.markdown("Feature Importance")
        # Try to get feature names from the UI/example (excluding target-like cols)
        drop_cols = [c for c in example_df.columns if c.lower() in {"credit_bad", "target", "y"}]
        feature_names = [c for c in example_df.columns if c not in drop_cols]

        fi_df = get_feature_importances(model, meta, feature_names)
        if fi_df is None or fi_df.empty:
            st.info("No feature importance available (neither in metadata nor model attributes).")
        else:
            # Show top k
            top_k = st.slider("Show top features", min_value=3, max_value=min(30, len(fi_df)), value=min(10, len(fi_df)))
            show_df = fi_df.head(top_k).copy()

            # Plot with matplotlib (horizontal bar)
            fig, ax = plt.subplots(figsize=(6, max(2, 0.35 * top_k)))
            ax.barh(show_df["feature"][::-1], show_df["importance"][::-1])
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.grid(True, axis="x", linestyle=":", alpha=0.5)
            st.pyplot(fig)

            with st.expander("View raw importance data"):
                st.dataframe(fi_df.reset_index(drop=True), use_container_width=True)

    with tab_history:
        st.markdown("Prediction History")
        ensure_history_state()

        if st.session_state.history_df.empty:
            st.info("No predictions yet. Generate a prediction in the **Prediction** tab.")
        else:
            # Filters
            with st.expander("Filter & Search"):
                # simple text filter on prediction label or columns
                search_text = st.text_input("Search text (matches anywhere in row)")
                df_hist = st.session_state.history_df.copy()
                if search_text:
                    mask = df_hist.apply(lambda row: row.astype(str).str.contains(search_text, case=False, na=False).any(), axis=1)
                    df_hist = df_hist[mask]

                # optional range filter on probability
                if "_prob_bad" in df_hist.columns and df_hist["_prob_bad"].notna().any():
                    pmin, pmax = float(df_hist["_prob_bad"].min()), float(df_hist["_prob_bad"].max())
                    if not np.isnan(pmin) and not np.isnan(pmax):
                        r = st.slider("Filter by probability of bad", 0.0, 1.0, (0.0, 1.0), 0.01)
                        df_hist = df_hist[(df_hist["_prob_bad"].fillna(-1) >= r[0]) & (df_hist["_prob_bad"].fillna(-1) <= r[1])]

            st.dataframe(df_hist, use_container_width=True, height=400)
            download_button_for_history()

            if not st.session_state.history_df.empty and st.button("Clear History"):
                st.session_state.history_df = pd.DataFrame()
                st.toast("History cleared.", icon="🧹")

    with tab_about:
        st.markdown("About this App")
        st.write(
            """
            - **Interactive inputs**: numeric sliders and categorical dropdowns are generated from your training schema or uploaded CSV.
            - **Custom threshold**: optionally override the decision boundary using the predicted probability (if available).
            - **Insights**: shows test metrics, CV summary, and feature importance (from metadata or model attributes).
            - **History**: every prediction is stored locally in session state and can be downloaded as CSV.
            - **Templates**: download a one-row CSV template that matches the current schema.
            """
        )
        with st.expander("Model & Metadata Info"):
            st.json({
                "model_type": type(model).__name__,
                "metadata_keys": sorted(list(meta.keys()))
            })

        st.caption("Tip: To re-train, run `python train_model.py --auto_download` or point to your CSV with `--data_csv`.")

if __name__ == "__main__":
    main()
