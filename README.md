IoT Architecture & Data Flow

1. Sensors: Deployed in wearable devices (e.g., wristbands with HRV/GSR), ambient sensors (temperature/humidity modules), plus activity trackers (step count).
2. Data Transmission: Typically, over Bluetooth Low Energy (BLE) or Wi-Fi to a gateway or cloud service. Data is timestamped and aggregated in a single pipeline.
3. Data Pipeline (On Edge or Cloud): Receives raw sensor data, runs the ML pipeline described below, and outputs an updated stress probability or class label.
4. User Feedback: The system can push alerts (notifications) when stress crosses a threshold, or store data for further analysis in a web-based dashboard.

Core Tools & Libraries

1. Python & Streamlit
2. Streamlit for an interactive UI (tabs: data prep, model training, interpretation).
3. Python ecosystem for robust data manipulation.
4. Scikit-learn: Outlier detection, random forests, partial dependence plots, permutation importance.
5. XGBoost: Gradient boosting for tabular classification.
6. Imbalanced-Learn (SMOTE): Corrects class imbalance by oversampling minority classes.
7. SHAP: Local explanation of predictions—vital for understanding stress triggers.
8. Plotly: Rich, interactive visualizations (e.g., confusion matrices).
