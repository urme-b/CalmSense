import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import shap

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from statsmodels.formula.api import ols

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import (permutation_importance,
                                PartialDependenceDisplay)
from imblearn.over_sampling import SMOTE

import time

def hybrid_outliers(df, contamination=0.02):
    """
    Removes outliers flagged by either:
      - IsolationForest
      - IQR-based approach
    'contamination' is fraction of outliers to expect for IsolationForest.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "stress_label" in numeric_cols:
        numeric_cols.remove("stress_label")
    if not numeric_cols:
        return df  # no numeric => skip

    # IsolationForest
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_preds = iso.fit_predict(df[numeric_cols])  # -1 => outlier

    # IQR
    iqr_flags = np.zeros(len(df), dtype=bool)
    for c in numeric_cols:
        col_data = df[c]
        q1, q3 = col_data.quantile([0.25, 0.75])
        iqr_val = q3 - q1
        lower = q1 - 1.5*iqr_val
        upper = q3 + 1.5*iqr_val
        out_mask = (col_data<lower)|(col_data>upper)
        iqr_flags = iqr_flags | out_mask

    combined = (iso_preds==-1) | iqr_flags
    new_df = df[~combined].reset_index(drop=True)
    return new_df

def add_noise(df, noise_pct=3.0):
    """
    Adds small Gaussian noise to 'temperature' and 'humidity'
    to simulate real sensor drift.
    """
    for col in ["temperature","humidity"]:
        if col in df.columns:
            mean_val = df[col].mean()
            sigma = (noise_pct/100.0)*mean_val
            df[col] += np.random.normal(0, sigma, len(df))
    return df

def impute_missing(df, strategy="median"):
    """
    Simple numeric imputation with chosen strategy. (median, mean, etc.)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    si = SimpleImputer(strategy=strategy)
    df[num_cols] = si.fit_transform(df[num_cols])
    return df

def shap_explainer(model, X_train_sub, X_test_sub):
    """
    Attempt TreeExplainer or fallback to KernelExplainer
    for multi-class => shap_vals is list => pick last or average
    """
    from shap import TreeExplainer, KernelExplainer
    try:
        expl = TreeExplainer(model)
        shap_vals = expl.shap_values(X_test_sub)
    except:
        expl = KernelExplainer(model.predict_proba, X_train_sub)
        shap_vals = expl.shap_values(X_test_sub, nsamples=100)
    return expl, shap_vals

def partial_dependence_multiclass(model, X_test, y_test, feature_names):
    """
    Multi-class friendly partial dependence.
    The user picks feature index & target class (if multi-class).
    """
    st.write("**Partial Dependence (Multi-Class Fix)**")

    unique_labels = np.unique(y_test)
    n_classes = len(unique_labels)
    if n_classes < 2:
        st.warning("Not enough classes for partial dependence.")
        return

    feat_idx = st.selectbox("Pick feature index", range(len(feature_names)), 
                            format_func=lambda i: feature_names[i])

    if n_classes>2:
        target_class = st.selectbox("Which class index to interpret?", range(n_classes),
                                    format_func=lambda c: f"Class {c}")
    else:
        target_class = 1  # default interpret positive class in binary

    fig, ax = plt.subplots()
    PartialDependenceDisplay.from_estimator(
        model, X_test, [feat_idx],
        target=target_class,
        ax=ax
    )
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="RefinedCalmSenseSubmission", layout="wide")

    tab1, tab2, tab3 = st.tabs([
        "Data & Prep",
        "Train & Evaluate",
        "Interpret & Real-Time"
    ])

    with tab1:
        st.title("1) Data & Preparation")
        csv_path = st.text_input("CSV Path", "Stress-Lysis.csv")

        # Load CSV
        if st.button("Load CSV"):
            df = pd.read_csv(csv_path)
            st.session_state["df"] = df
            st.success(f"Loaded {len(df)} rows (shape={df.shape}).")
            st.dataframe(df.head())

        # If data is loaded
        if "df" in st.session_state:
            df = st.session_state["df"]

            st.subheader("Basic Renaming & Drop NA")
            rename_map = {
                "Humidity": "humidity",
                "Temperature": "temperature",
                "Step_count": "steps",
                "Stress_Level": "stress_label"
            }
            df.rename(columns=rename_map, inplace=True, errors="ignore")
            df.dropna(inplace=True)

            # ensure stress_label is int
            if "stress_label" in df.columns and df["stress_label"].dtype == object:
                try:
                    df["stress_label"] = df["stress_label"].astype(int)
                except ValueError:
                    st.error("Could not convert stress_label to int. Check CSV data.")

            st.write("Data after rename & dropna =>", df.head(), " shape =>", df.shape)

            # Noise injection
            if st.checkbox("Add Noise to temp/humidity?", value=True):
                noise_lvl = st.slider("Noise level (%)", 1,10,3)
                df = add_noise(df, noise_lvl)

            # Outliers
            if st.button("Hybrid Outlier Removal"):
                before = len(df)
                df = hybrid_outliers(df, 0.02)
                st.write(f"Dropped {before - len(df)} outliers => shape={df.shape}")

            # missing
            if st.checkbox("Impute Missing (median)?", value=True):
                df = impute_missing(df, "median")
                st.write("Imputed => shape=", df.shape)

            st.session_state["df"] = df
            st.write("**Final Data** =>", df.head())
            st.success("Data prep done. Move to 'Train & Evaluate' tab.")

    with tab2:
        st.title("2) Train & Evaluate")
        if "df" not in st.session_state:
            st.warning("No data. Go to 'Data & Prep' tab.")
        else:
            df = st.session_state["df"]

            # optional feature engineering
            if st.checkbox("Create interaction (humidity * temperature)?"):
                if "humidity" in df.columns and "temperature" in df.columns:
                    df["hum_temp_interact"] = df["humidity"] * df["temperature"]

            # define features
            feats_found = []
            for c in ["humidity","temperature","steps","hum_temp_interact"]:
                if c in df.columns:
                    feats_found.append(c)
            st.write("Features =>", feats_found)

            if "stress_label" not in df.columns:
                st.error("No 'stress_label' column found. Stopping.")
                st.stop()

            X = df[feats_found].values
            y = df["stress_label"].values

            test_size_pct = st.slider("Test size (%)", 5,50,20)/100.0
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_pct, random_state=42, stratify=y
            )

            if st.checkbox("SMOTE for imbalance?"):
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)

            # scale
            scl = StandardScaler()
            X_train_s = scl.fit_transform(X_train)
            X_test_s  = scl.transform(X_test)

            # pick models
            run_rf  = st.checkbox("RandomForest", value=True)
            run_xgb = st.checkbox("XGBoost", value=True)
            run_svm = st.checkbox("SVM", value=True)

            if st.button("Train"):
                models = {}
                if run_rf:
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_train_s, y_train)
                    models["RF"] = rf
                if run_xgb:
                    xgb_model = xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42)
                    xgb_model.fit(X_train_s, y_train)
                    models["XGB"] = xgb_model
                if run_svm:
                    svm = SVC(probability=True, kernel="rbf", random_state=42)
                    svm.fit(X_train_s, y_train)
                    models["SVM"] = svm

                # Evaluate
                best_name = None
                best_acc  = 0
                for name, mdl in models.items():
                    y_pred = mdl.predict(X_test_s)
                    acc  = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
                    f1_  = f1_score(y_test, y_pred, average="macro", zero_division=0)
                    st.write(f"{name} => ACC={acc*100:.2f}%, Prec={prec:.2f}, Rec={rec:.2f}, F1={f1_:.2f}")

                    # confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = ff.create_annotated_heatmap(z=cm,
                                                         x=[str(i) for i in np.unique(y)],
                                                         y=[str(i) for i in np.unique(y)],
                                                         colorscale="Blues")
                    st.plotly_chart(fig_cm, use_container_width=True)

                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))

                    if acc>best_acc:
                        best_acc = acc
                        best_name= name

                st.success(f"Best model => {best_name}, ACC ~ {best_acc*100:.2f}%")

                # store
                st.session_state["best_model"]   = models[best_name]
                st.session_state["best_model_nm"]= best_name
                st.session_state["X_test_s"]     = X_test_s
                st.session_state["y_test"]       = y_test
                st.session_state["scaler"]       = scl
                st.session_state["feats"]        = feats_found

            # Permutation
            if "best_model" in st.session_state and st.checkbox("Permutation Importance?"):
                best_mdl = st.session_state["best_model"]
                X_te_s   = st.session_state["X_test_s"]
                y_te     = st.session_state["y_test"]
                feats    = st.session_state["feats"]

                st.write("Computing permutation importance (might be slow).")
                r = permutation_importance(best_mdl, X_te_s, y_te, n_repeats=5, random_state=42)
                means = r.importances_mean
                stds  = r.importances_std
                idx = np.argsort(means)
                fig_imp, ax_imp = plt.subplots()
                ax_imp.barh(np.array(feats)[idx], means[idx], xerr=stds[idx], color="teal")
                ax_imp.set_title("Permutation Importance (mean decrease in ACC)")
                st.pyplot(fig_imp)

    with tab3:
        st.title("3) Interpretation & Real-Time")
        if "best_model" not in st.session_state:
            st.warning("No model. Go to 'Train & Evaluate' tab.")
        else:
            best_mdl = st.session_state["best_model"]
            X_te_s   = st.session_state["X_test_s"]
            y_te     = st.session_state["y_test"]
            feats    = st.session_state["feats"]

            # Partial dependence
            if st.checkbox("Partial Dependence Plot?"):
                partial_dependence_multiclass(best_mdl, X_te_s, y_te, feats)

            # SHAP
            if st.checkbox("SHAP Explanation?"):
                st.write("We do a small subset for speed.")
                sub_sz = st.slider("Subset size", 10, min(len(X_te_s),200), 50)
                X_te_sub = X_te_s[:sub_sz]

                expl, shap_vals = shap_explainer(best_mdl, X_te_sub, X_te_sub)
                if isinstance(shap_vals, list) and len(shap_vals)>1:
                    st.write("Multi-class => picking last class for demonstration.")
                    shap_array = shap_vals[-1]
                else:
                    shap_array = shap_vals

                shap.initjs()
                st.write("**SHAP Summary (Beeswarm)**")
                fig_shap, ax_shap = plt.subplots()
                shap.summary_plot(shap_array, X_te_sub, feature_names=feats, show=False)
                st.pyplot(fig_shap)

                st.write("**SHAP Bar Plot**")
                fig_shap2, ax_shap2 = plt.subplots()
                shap.summary_plot(shap_array, X_te_sub, feature_names=feats, plot_type="bar", show=False)
                st.pyplot(fig_shap2)

            # Real-time sim
            if "df" not in st.session_state:
                st.warning("No final df stored. We'll do a placeholder simulation.")
            else:
                if st.button("Simulate Last 10 Rows"):
                    df = st.session_state["df"]
                    feats = st.session_state["feats"]
                    tail_df = df.tail(10).reset_index(drop=True)
                    results = []
                    for i in range(len(tail_df)):
                        row = tail_df.loc[i, feats].values.reshape(1,-1)
                        row_scl = st.session_state["scaler"].transform(row)
                        pred = best_mdl.predict(row_scl)[0]
                        row_info = {c: tail_df.loc[i,c] for c in feats}
                        row_info["Row #"] = i+1
                        row_info["Prediction"] = pred
                        results.append(row_info)
                        time.sleep(0.2)
                    st.write(pd.DataFrame(results))

            st.success("Interpretation & Real-Time done. Pipeline complete.")

if __name__ == "__main__":
    main()