import streamlit as st
import pandas as pd
import numpy as np
import builtins
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
import io
import time
import random
import string
import os
from scipy.stats import wasserstein_distance

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    ._profileContainer_gzau3_53 {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
# Ensure compatibility with DataSynthesizer
builtins.np = np

# Page config
st.set_page_config(page_title="Synthetic Data Generator", page_icon="📊", layout="wide")
st.title("📊 Synthetic Data Studio (DataSynthesizer)")

# Function to generate random short code
def random_code(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# ===== Upload Section =====
st.subheader("📂 Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Upload CSV or CSV.GZ file",
    type=["csv", "gz"]
)

if uploaded_file is not None:
    try:
        # Read dataset
        if uploaded_file.name.endswith(".gz"):
            df = pd.read_csv(uploaded_file, compression="gzip")
        else:
            df = pd.read_csv(uploaded_file)

        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

        # Show uploaded dataset in expandable section
        with st.expander("🔍 Preview Uploaded Dataset", expanded=True):
            st.dataframe(df.head())

        # Generate unique code for filenames
        code = random_code()
        base_name = os.path.splitext(uploaded_file.name)[0]
        uploaded_filename = f"{base_name}_{code}.csv"
        df.to_csv(uploaded_filename, index=False)
        st.info(f"📂 Uploaded file saved as: `{uploaded_filename}`")

        # ===== Backend Settings (hidden from user) =====
        epsilon = 1.0  # Ideal privacy parameter
        k = 2          # Ideal max parents in Bayesian Network

        # ===== User Setting =====
        with st.expander("⚙️ Settings", expanded=True):
            n = st.number_input(
                "Number of synthetic rows",
                min_value=100,
                max_value=100000,
                value=2500,
                help="Set the number of synthetic rows to generate (max 100,000)."
            )

        # ===== Generate Button =====
        if st.button("🚀 Generate Synthetic Data"):
            # Dataset description
            with st.spinner("🔄 Generating dataset description..."):
                describer = DataDescriber()
                describer.describe_dataset_in_correlated_attribute_mode(
                    dataset_file=uploaded_filename,
                    epsilon=epsilon,
                    k=k,
                    attribute_to_is_categorical={},
                    attribute_to_is_candidate_key={}
                )
                describer.save_dataset_description_to_file("data_description.json")
                time.sleep(1)
            st.success("✅ Dataset description created")

            # Generate synthetic dataset
            with st.spinner("🔄 Generating synthetic dataset..."):
                generator = DataGenerator()
                generator.generate_dataset_in_correlated_attribute_mode(
                    n=n,
                    description_file="data_description.json"
                )
                synthetic_filename = f"synthetic_{code}.csv"
                generator.save_synthetic_data(synthetic_filename)
                time.sleep(1)

            # Load and show synthetic dataset
            synthetic_df = pd.read_csv(synthetic_filename)
            st.success(f"🎉 Synthetic dataset generated successfully! (Saved as `{synthetic_filename}`)")

            with st.expander("🔍 Preview Synthetic Dataset", expanded=True):
                st.dataframe(synthetic_df.head())

            # ===== Overall Accuracy Calculation =====
            st.subheader("📊 Overall Accuracy Comparison")

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()

            accuracy_scores = []

            # Numeric columns: using Wasserstein distance
            for col in numeric_cols:
                try:
                    dist = wasserstein_distance(df[col], synthetic_df[col])
                    score = max(0, 1 - dist / (df[col].max() - df[col].min()))
                    accuracy_scores.append(score)
                except Exception:
                    pass

            # Categorical columns: using normalized frequency similarity
            for col in categorical_cols:
                try:
                    real_counts = df[col].value_counts(normalize=True)
                    synth_counts = synthetic_df[col].value_counts(normalize=True)
                    combined = pd.concat([real_counts, synth_counts], axis=1).fillna(0)
                    score = 1 - (combined.diff(axis=1).abs().sum(axis=1).mean() / 2)  # scaled 0-1
                    accuracy_scores.append(score)
                except Exception:
                    pass

            if accuracy_scores:
                overall_accuracy = round(np.mean(accuracy_scores) * 100, 2)
                st.metric("🌟 Overall Accuracy (%)", f"{overall_accuracy}%")
            else:
                st.info("No comparable columns found for accuracy calculation.")

            # ===== Download Button =====
            csv_buffer = io.StringIO()
            synthetic_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Synthetic Dataset",
                data=csv_buffer.getvalue(),
                file_name=synthetic_filename,
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
