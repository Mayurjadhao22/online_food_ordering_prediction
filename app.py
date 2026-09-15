import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Response Predictor",
    page_icon="🔮",
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load the trained RandomForest model from model.pkl."""
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found in the root directory.")
        st.stop()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def main():
    st.title("🔮 Customer Response Predictor")
    st.write(
        "Enter customer demographic and socioeconomic details below to predict their response."
    )

    model = load_model()

    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=30, step=1
        )
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        marital_status = st.selectbox(
            "Marital Status", options=["Single", "Married", "Divorced", "Widowed"]
        )
        occupation = st.selectbox(
            "Occupation",
            options=["Student", "Employee", "Self Employed", "Unemployed", "Retired"],
        )

    with col2:
        monthly_income = st.selectbox(
            "Monthly Income",
            options=[
                "No Income",
                "Below 10,000",
                "10,001 to 25,000",
                "25,001 to 50,000",
                "More than 50,000",
            ],
        )
        educational_qualifications = st.selectbox(
            "Educational Qualifications",
            options=["School", "Under Graduate", "Post Graduate", "Ph.D"],
        )
        family_size = st.number_input(
            "Family Size", min_value=1, max_value=20, value=3, step=1
        )
        customer_type = st.selectbox(
            "Customer Type", options=["New", "Existing", "Frequent"]
        )

    # Encode categorical features into numeric formats standard for ML pipelines
    # Adjust mappings below if your training pipeline used a different ordinal sequence
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
    occupation_map = {
        "Student": 0,
        "Employee": 1,
        "Self Employed": 2,
        "Unemployed": 3,
        "Retired": 4,
    }
    income_map = {
        "No Income": 0,
        "Below 10,000": 1,
        "10,001 to 25,000": 2,
        "25,001 to 50,000": 3,
        "More than 50,000": 4,
    }
    edu_map = {"School": 0, "Under Graduate": 1, "Post Graduate": 2, "Ph.D": 3}
    cust_type_map = {"New": 0, "Existing": 1, "Frequent": 2}

    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender_map[gender],
                "Marital Status": marital_map[marital_status],
                "Occupation": occupation_map[occupation],
                "Monthly Income": income_map[monthly_income],
                "Educational Qualifications": edu_map[
                    educational_qualifications
                ],
                "Family size": family_size,
                "Customer Type": cust_type_map[customer_type],
            }
        ]
    )

    st.markdown("---")

    if st.button("Predict Customer Response", type="primary"):
        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]

            st.subheader("Prediction Result")
            if prediction == "Yes" or prediction == 1:
                st.success(f"**Result:** Positive Response ({prediction})")
            else:
                st.warning(f"**Result:** Negative Response ({prediction})")

            # Display prediction probabilities if available
            if hasattr(model, "classes_"):
                st.write("**Prediction Probabilities:**")
                prob_df = pd.DataFrame(
                    [probabilities], columns=[str(c) for c in model.classes_]
                )
                st.dataframe(prob_df.style.format("{:.2%}"))

        except Exception as e:
            st.error(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()
