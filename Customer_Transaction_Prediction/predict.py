import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os


FOLDER = "ct_models"

model = joblib.load(os.path.join(FOLDER, "best_customer_model.pkl"))
scaler = joblib.load(os.path.join(FOLDER, "scaler.pkl"))
var = joblib.load(os.path.join(FOLDER, "variance_filter.pkl"))
top_features = np.load(os.path.join(FOLDER, "top_features.npy"), allow_pickle=True)

#  PREDICTION FUNCTION 
def predict_transaction(new_df):
    """
    Predict transaction likelihood for new customers.
    new_df: DataFrame with feature columns only
    """
    # Apply variance threshold
    X_vt = var.transform(new_df)

    # Select top features
    if isinstance(top_features[0], int):
        X_sel = X_vt[:, top_features]  # if top_features are indices
    else:
        X_sel = X_vt[:, :len(top_features)]

    # Scaling
    X_scaled = scaler.transform(X_sel)

    # Prediction
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)

    # Return dataframe with predictions
    df_out = new_df.copy()
    df_out["transaction_probability"] = probs
    df_out["prediction"] = preds
    return df_out

#  STREAMLIT UI 
st.title("Customer Transaction Prediction App")
st.write("Predict whether a customer will make a transaction.")

#  1. EXISTING CLIENTS
st.header("📌 Predict for Existing Customers")

if st.button("Run Prediction for Existing Clients"):
    df = pd.read_csv("train.csv")
    df_features = df.drop(columns=["ID_code", "target"])
    results = predict_transaction(df_features)
    results["ID_code"] = df["ID_code"]
    results["target"] = df["target"]

    st.success("Prediction complete!")
    st.write(results.head())

    st.download_button(
        "Download Predictions",
        data=results.to_csv(index=False),
        file_name="existing_customer_predictions.csv"
    )

#  2. NEW CLIENTS
st.header("📌 Predict for New Customers")

uploaded = st.file_uploader("Upload a CSV with var_0…var_199", type=["csv"])

if uploaded is not None:
    new_df = pd.read_csv(uploaded)
    st.write("Preview of uploaded data:")
    st.write(new_df.head())

    if st.button("Predict for New Customers"):
        output = predict_transaction(new_df)
        st.success("Prediction complete!")
        st.write(output.head())

        st.download_button(
            "Download Results",
            data=output.to_csv(index=False),
            file_name="new_customer_predictions.csv"
        )

# 3. MANUAL
st.header("📌 Manual Customer Entry (Single Prediction)")

if st.checkbox("Enter one customer manually"):
    input_data = {}
    # Show first 10 vars for UI; user can expand
    for i in range(10):
        col = f"var_{i}"
        input_data[col] = st.number_input(col, value=0.0)

    if st.button("Predict Manually"):
        df_manual = pd.DataFrame([input_data])
        result = predict_transaction(df_manual)
        prob = float(result["transaction_probability"][0])
        pred = int(result["prediction"][0])

        st.write(f"📌 **Transaction Probability:** {prob:.4f}")
        st.write(f"📌 **Prediction:** {'Yes' if pred==1 else 'No'}")
