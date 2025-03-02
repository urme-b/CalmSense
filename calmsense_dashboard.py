import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, time

from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from statsmodels.formula.api import ols

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb

sns.set_style("whitegrid")

CSV_PATH = "/Users/urmebose/Documents/CalmSense/Stress-Lysis.csv"


def load_data():
    """Load CSV, show preview and shape."""
    st.title("CalmSense")
    st.header("1) Load Data")
    try:
        df = pd.read_csv(CSV_PATH)
        st.success(f"CSV loaded from: {CSV_PATH}")
    except FileNotFoundError:
        st.error(f"Cannot find CSV at {CSV_PATH}")
        st.stop()

    st.write("**Preview**:", df.head())
    st.write("**Shape**:", df.shape)

    buf = io.StringIO()
    df.info(buf=buf)
    st.text(buf.getvalue())
    return df

def clean_and_rename(df):
    """Step 2: Drop NaNs, rename columns, ensure stress_label is numeric."""
    st.header("2) Clean & Rename")
    rename_map = {
        "Humidity": "humidity",
        "Temperature": "temperature",
        "Step_count": "steps",
        "Stress_Level": "stress_label"
    }
    df.rename(columns=rename_map, inplace=True, errors="ignore")
    st.write("Renamed columns:", list(df.columns))

    before = len(df)
    df.dropna(inplace=True)
    st.write(f"Dropped {before - len(df)} rows => shape={df.shape}")

    if "stress_label" in df.columns:
        if df["stress_label"].dtype == object:
            try:
                df["stress_label"] = df["stress_label"].astype(int)
            except ValueError:
                st.error("Could not convert 'stress_label' to int.")
    else:
        st.warning("No 'stress_label' column found.")
    return df

def add_noise_option(df):
    """
    Optional Step: inject random noise into temperature/humidity
    """
    st.header("Noise")
    do_noise = st.checkbox("Noise (temp/humidity)?", value=True)
    if do_noise:
        noise_level = st.slider("Noise Level (%)", 1, 10, 3)
        if "humidity" in df.columns:
            mean_h = df["humidity"].mean()
            df["humidity"] += np.random.normal(0, (noise_level/100)*mean_h, len(df))
        if "temperature" in df.columns:
            mean_t = df["temperature"].mean()
            df["temperature"] += np.random.normal(0, (noise_level/100)*mean_t, len(df))
        st.write(f"Injected ~{noise_level}% noise.")
    else:
        st.info("Skipping noise.")
    return df

def advanced_stats(df):
    """Step 3: Box plots, T-test, ANOVA, correlation heatmap, etc."""
    st.header("3) Advanced Stats & Visuals")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Box Plot
    st.subheader("Box Plot")
    if num_cols:
        fig, ax = plt.subplots(figsize=(8,5))
        melt_df = df[num_cols].melt(var_name="Feature", value_name="Value")
        sns.boxplot(x="Feature", y="Value", data=melt_df, palette="Set2", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No numeric => skip box plot.")

    # T-test => temperature vs humidity
    if all(x in num_cols for x in ["temperature","humidity"]):
        t_stat, p_val = ttest_ind(df["temperature"], df["humidity"], equal_var=False)
        st.write(f"T-test (temp vs humidity): t={t_stat:.3f}, p={p_val:.3f}")
    else:
        st.info("Skipping T-test => missing 'temperature'/'humidity'.")

    # ANOVA => temperature across stress_label
    if "stress_label" in df.columns and "temperature" in num_cols:
        unique_lbls = df["stress_label"].unique()
        if len(unique_lbls) > 1:
            groups = [df["temperature"][df["stress_label"]==lbl] for lbl in unique_lbls]
            f_stat, p_anova = f_oneway(*groups)
            st.write(f"ANOVA => F={f_stat:.3f}, p={p_anova:.3f}")
        else:
            st.info("Not enough distinct stress_label groups for ANOVA.")
    else:
        st.info("Skipping ANOVA => need 'stress_label' + 'temperature'")

    # Correlation
    if len(num_cols) > 1:
        st.write("**Correlation Matrix**")
        corr = df[num_cols].corr()
        st.write(corr)
        fig2, ax2 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("Not enough numeric columns => skip correlation.")

def remove_outliers(df):
    """Step 4: IsolationForest outlier removal (2% contamination)."""
    st.header("4) Outlier Removal")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "stress_label" in num_cols:
        num_cols.remove("stress_label")

    if not num_cols:
        st.info("No numeric => skip outlier detection.")
        return df

    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(df[num_cols])
    preds = iso.predict(df[num_cols])  # -1 => outlier
    before = len(df)
    df = df[preds==1].reset_index(drop=True)
    st.write(f"Removed {before - len(df)} outliers => shape={df.shape}")
    return df

def feature_engineering(df):
    """Step 5: e.g., hum_temp_interact = humidity * temperature."""
    st.header("5) Feature Engineering")
    if "humidity" in df.columns and "temperature" in df.columns:
        df["hum_temp_interact"] = df["humidity"] * df["temperature"]
        st.write("Created 'hum_temp_interact' = humidity * temperature")
    else:
        st.warning("Missing 'humidity'/'temperature' => skip interact.")
    st.write("Shape =>", df.shape)
    return df

def split_and_scale(df):
    """Step 6: Train/test split & standard scaling."""
    st.header("6) Split & Scale")
    needed = ["humidity","temperature","steps","stress_label"]
    for c in needed:
        if c not in df.columns:
            st.error(f"Missing '{c}' => can't proceed.")
            return None

    features = ["humidity","temperature","steps"]
    if "hum_temp_interact" in df.columns:
        features.append("hum_temp_interact")

    X = df[features].values
    y = df["stress_label"].values

    if len(df) < 5:
        st.error("Too few rows => can't split.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    st.write(f"Train size={len(X_train)}, Test size={len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, features

def train_models(X_train, y_train):
    """Step 7: RandomForest & XGBoost hyperparam tuning via RandomizedSearchCV."""
    st.header("7) Model Tuning (RandomForest + XGBoost)")

    # RandomForest
    st.subheader("RandomForest Search")
    rf_params = {
        "n_estimators": [50,100],
        "max_depth": [3,5,None],
        "min_samples_split": [2,5],
        "min_samples_leaf": [1,2]
    }
    rf = RandomForestClassifier(random_state=42)
    rf_search = RandomizedSearchCV(
        rf, rf_params, n_iter=4, cv=3, scoring="accuracy", n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    st.write("RF best params:", rf_search.best_params_)
    st.write("RF CV accuracy:", rf_search.best_score_)

    # XGBoost
    st.subheader("XGBoost Search")
    xgb_model = xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42)
    xgb_params = {
        "n_estimators": [50,100],
        "max_depth": [3,5],
        "learning_rate": [0.1,0.05]
    }
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_params, n_iter=4, cv=3, scoring="accuracy", n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    st.write("XGB best params:", xgb_search.best_params_)
    st.write("XGB CV accuracy:", xgb_search.best_score_)

    return best_rf, best_xgb

def evaluate_models(rf_best, xgb_best, X_test, y_test):
    """
    Step 8: Evaluate each model on test set, pick best by accuracy,
    display confusion matrix & classification report, final metric.
    """
    st.header("8) Evaluate & Pick Best")

    models = [rf_best, xgb_best]
    names  = ["RandomForest","XGBoost"]

    best_acc = 0
    best_model = None
    best_name = None
    results = []
    for mdl, nm in zip(models, names):
        y_pred = mdl.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append({
            "Model": nm,
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec:.2f}",
            "Recall": f"{rec:.2f}",
            "F1": f"{f1:.2f}"
        })
        if acc > best_acc:
            best_acc = acc
            best_model = mdl
            best_name = nm

    st.write("**Test-Set Metrics**")
    st.table(pd.DataFrame(results))

    for mdl, nm in zip(models, names):
        st.subheader(f"{nm} => Confusion Matrix & Classification Report")
        y_pred = mdl.predict(X_test)
        st.write(confusion_matrix(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

    st.success(f"Best model is {best_name}, accuracy={best_acc*100:.2f}%")
    st.metric(label="Final Best Model Accuracy", value=f"{best_acc*100:.2f}%")
    return best_model, best_name

def real_time_simulation(model, df, scaler, features):
    """
    Step 9: Predict on last 10 rows => simulate near-real-time usage.
    """
    st.header("9) Real-Time Simulation")
    tail_df = df.tail(10).reset_index(drop=True)
    if tail_df.empty:
        st.warning("No data for simulation.")
        return

    X_sim = tail_df[features].values
    X_sim_scaled = scaler.transform(X_sim)
    label_map = {0:"Low Stress",1:"Normal Stress",2:"High Stress"}

    sim_out = []
    for i, row_data in enumerate(X_sim_scaled):
        pred_label = model.predict([row_data])[0]
        row_dict = {f: tail_df.loc[i,f] for f in features if f in tail_df.columns}
        row_dict["Row #"] = i+1
        row_dict["Prediction"] = label_map.get(pred_label, pred_label)
        sim_out.append(row_dict)
        time.sleep(0.3)

    st.write(pd.DataFrame(sim_out))

def feature_importance_analysis(model, X_test, y_test, feature_names):
    """
    Step 10: Permutation Feature Importance (scikit-learn).
    Creates a bar plot showing each feature's mean importance.
    """
    st.header("10) Permutation Feature Importance")

    from sklearn.inspection import permutation_importance
    results = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, scoring="accuracy"
    )
    importances_mean = results.importances_mean
    importances_std  = results.importances_std
    indices = np.argsort(importances_mean)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(
        [feature_names[i] for i in indices],
        importances_mean[indices],
        xerr=importances_std[indices],
        color="teal", ecolor="black"
    )
    ax.set_xlabel("Mean Importance (decrease in accuracy)")
    ax.set_title("Permutation Feature Importance")
    st.pyplot(fig)

def main():
    df = load_data()
    if df is None:
        return

    df = clean_and_rename(df)
    if df.empty:
        st.stop()

    df = add_noise_option(df)
    advanced_stats(df)

    df = remove_outliers(df)
    if df.empty:
        st.error("All data removed => stopping.")
        st.stop()

    df = feature_engineering(df)
    if df.empty:
        st.error("No data => stopping.")
        st.stop()

    splitted = split_and_scale(df)
    if splitted is None:
        st.stop()
    X_train_s, X_test_s, y_train, y_test, scaler, features = splitted

    rf_best, xgb_best = train_models(X_train_s, y_train)
    best_model, best_name = evaluate_models(rf_best, xgb_best, X_test_s, y_test)

    real_time_simulation(best_model, df, scaler, features)

    # Permutation Importance
    feature_importance_analysis(best_model, X_test_s, y_test, features)

if __name__ == "__main__":
    main()