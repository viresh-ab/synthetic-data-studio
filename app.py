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

# Ensure compatibility with DataSynthesizer
builtins.np = np

st.set_page_config(page_title="Synthetic Data Generator", page_icon="📊", layout="wide")
st.title("📊 Synthetic Data Studio (DataSynthesizer)")

# Function to generate random short code
def random_code(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or CSV.GZ file", type=["csv", "gz"])

if uploaded_file is not None:
    try:
        # Read dataset
        if uploaded_file.name.endswith(".gz"):
            df = pd.read_csv(uploaded_file, compression="gzip")
        else:
            df = pd.read_csv(uploaded_file)

        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

        st.write("### 🔍 Sample of Uploaded Dataset")
        st.dataframe(df.head())

        # Generate unique code for this run
        code = random_code()
        
        # Save uploaded file locally with code
        base_name = os.path.splitext(uploaded_file.name)[0]
        uploaded_filename = f"{base_name}_{code}.csv"
        df.to_csv(uploaded_filename, index=False)

        st.info(f"📂 Uploaded file saved as: `{uploaded_filename}`")

        # Parameters for user input
        st.sidebar.header("⚙️ Settings")
        epsilon = st.sidebar.slider("Privacy parameter (epsilon)", 0.1, 5.0, 1.0)
        k = st.sidebar.slider("Max parents in Bayesian Network (k)", 1, 5, 2)
        n = st.sidebar.number_input("Number of synthetic rows", 100, 10000, 2500)

        if st.button("🚀 Generate Synthetic Data"):
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

            with st.spinner("🔄 Generating synthetic dataset..."):
                generator = DataGenerator()
                generator.generate_dataset_in_correlated_attribute_mode(
                    n=n,
                    description_file="data_description.json"
                )
                synthetic_filename = f"synthetic_{code}.csv"
                generator.save_synthetic_data(synthetic_filename)
                time.sleep(1)

            # Load synthetic dataset
            synthetic_df = pd.read_csv(synthetic_filename)

            st.success(f"🎉 Synthetic dataset generated successfully! (Saved as `{synthetic_filename}`)")

            st.write("### 🔍 Sample of Synthetic Dataset")
            st.dataframe(synthetic_df.head())

            # Download button
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
