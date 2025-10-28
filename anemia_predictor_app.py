import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Configure Streamlit page
st.set_page_config(page_title="Anemia Prediction App", layout="wide")

file_path = r"C:\Users\Celine\Desktop\Research proposal\df_processed.csv"

try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["1. Introduction", "2. Data Visualization", "3. Predict Anemia"])

# Section 1: Introduction
if selection.startswith("1"):
    st.title("🎯 Introduction to the Anemia Prediction Model")
    st.markdown("""
    This application helps predict **anemia status among women** based on demographic and clinical features.

    - **Target variable**: Anemia status (Anemic or Not Anemic)
    - **Model**: Random Forest (best tuned model)
    - **Data**: Rwanda health demographic survey(RHDS)
    ---
    **Objectives:**
    1. Explore key insights from the data
    2. Predict anemia based on input factors
    3. Support data-driven decision-making
    """)

# Section 2: Data Visualization

# Section 2: Data Visualization
elif selection.startswith("2"):
    st.markdown(" <h3> Visualizations of Anemia Levels by Different Characteristics </h3>", unsafe_allow_html=True)

    # Load the cleaned dataset
    df = pd.read_excel(r"C:\Users\Celine\Desktop\Research proposal\df_cleaned_final.xlsx")

    # Define custom colors
    colors_palette = ["#87CEEB", "#ADD8E6", "#4682B4"]  # sky blue, light blue, ocean

    # --- First Row: 3 Charts ---
    col1, col2, col3 = st.columns(3)

    with col1:
        anemia_counts = df['anemia level'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(anemia_counts.values, labels=anemia_counts.index, autopct='%1.1f%%',
               startangle=45, colors=colors_palette[:len(anemia_counts)])
        ax.set_title("Overall Anemia Status")
        ax.legend(fontsize=6)
        st.pyplot(fig)

    with col2:
        anemia_region = pd.crosstab(df['region'], df['anemia level'])
        fig, ax = plt.subplots(figsize=(3, 2))
        anemia_region.plot(kind='bar', stacked=True, ax=ax, color=colors_palette, legend=False)  # disable pandas legend
        ax.set_title("Region", pad=10)  # move title slightly above
        ax.legend(anemia_region.columns, fontsize=6)  # manual legend with small font
        ax.set_xlabel("")  # optional: remove x-axis label for cleaner look
        st.pyplot(fig)

    with col3:
        anemia_wealth = pd.crosstab(df['wealth index'], df['anemia level'])
        fig, ax = plt.subplots(figsize=(3, 2))
        anemia_wealth.plot(kind='bar', stacked=True, ax=ax, color=colors_palette, legend=False)
        ax.set_title("Wealth Index", pad=10)
        ax.legend(anemia_wealth.columns, fontsize=6)
        ax.set_xlabel("")
        st.pyplot(fig)

    # --- Second Row: 3 Charts ---
    col4, col5, col6 = st.columns(3)
    with col4:

        anemia_edu = pd.crosstab(df['highest educational level'], df['anemia level'])
        fig, ax = plt.subplots(figsize=(3, 2))
        anemia_edu.plot(kind='bar', stacked=True, ax=ax, color=colors_palette, legend=False)
        ax.set_title("Education Level", pad=10)  # title above
        ax.legend(anemia_edu.columns, fontsize=6)  # small font legend
        ax.set_xlabel("")  # clean x-axis
        st.pyplot(fig)

    with col5:
        anemia_bmi = pd.crosstab(df['body mass index'], df['anemia level'])
        fig, ax = plt.subplots(figsize=(3, 2))
        anemia_bmi.plot(kind='bar', stacked=True, ax=ax, color=colors_palette, legend=False)
        ax.set_title("BMI Category", pad=10)
        ax.legend(anemia_bmi.columns, fontsize=6)
        ax.set_xlabel("")
        st.pyplot(fig)

    with col6:
        fig, ax = plt.subplots(figsize=(3, 3))
        for i, label in enumerate(df['anemia level'].unique()):
            subset = df[df['anemia level'] == label]
            ax.hist(subset["respondent's current age"], bins=15, alpha=0.6,
                label=label, color=colors_palette[i % len(colors_palette)])
        ax.set_title("Age by Anemia Status", pad=10)  # title above
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6)  # small font legend
        st.pyplot(fig)


    # --- Third Row: 2 Charts ---
    col7, col8, _ = st.columns([1,1,1])  # 2 charts centered

    with col7:
        anemia_contra = pd.crosstab(df['use of contraceptive method'], df['anemia level'])
        fig, ax = plt.subplots(figsize=(3, 2))
        anemia_contra.plot(kind='bar', stacked=True, ax=ax, color=colors_palette, legend=True)  # show legend
        ax.set_title("Contraceptive Use")  # title above
        ax.legend(fontsize=6)  # small legend font
        st.pyplot(fig)

    with col8:
        anemia_preg = pd.crosstab(df['currently pregnant'], df['anemia level'])
        fig, ax = plt.subplots(figsize=(3, 2))
        anemia_preg.plot(kind='bar', stacked=True, ax=ax, color=colors_palette, legend=False)
        ax.set_title("Pregnancy Status", pad=10)
        ax.legend(anemia_preg.columns, fontsize=6)
        st.pyplot(fig)


# Section 3: Predict

# Section 3: Predict
elif selection.startswith("3"): 
    st.title("🩺 Predict Anemia Among Women")
    st.info("Enter health information below to predict anemia status using the trained model:")

    try:
        # Load trained model + artifacts
        model_path = r"C:\Users\Celine\Documents\Hirwa documents\researchproject code\Best_Model_artifact\rf_best_model.pkl"
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)

        model = artifacts["model"]
        selected_features = artifacts["features"]
        le_target = artifacts["le_target"]

        # === User Inputs ===
        st.subheader("Enter patient information")

        input_data = {}

        # ===== Categorical SelectBoxes with mapping =====
        # Region / Province
        region_options = ["East", "Kigali City", "North", "South", "West"]
        selected_region = st.selectbox("Region / Province", options=region_options, index=0)
        region_mapping = {name: i for i, name in enumerate(region_options)}
        input_data['region'] = region_mapping[selected_region]

        # Place of residence
        residence_options = ["Rural", "Urban"]
        selected_residence = st.selectbox("Place of Residence", options=residence_options, index=0)
        residence_mapping = {name: i for i, name in enumerate(residence_options)}
        input_data['place of residence'] = residence_mapping[selected_residence]

        # Women's age group
        age_options = ["15-24", "25-34", "35-49"]
        selected_age = st.selectbox("Women Age", options=age_options, index=0)
        age_mapping = {name: i for i, name in enumerate(age_options)}
        input_data["respondent's current age"] = age_mapping[selected_age]

        # Religion
        #religion_options = ["Catholic", "Muslim", "Protestant", "Other"]
        #selected_religion = st.selectbox("Religion", options=religion_options, index=0)
        #religion_mapping = {name: i for i, name in enumerate(religion_options)}
        #input_data['religion'] = religion_mapping[selected_religion]

        # Women education
        women_edu_options = ["No Education", "Higher", "Secondary", "Primary"]
        selected_women_edu = st.selectbox("Women's Education Level", options=women_edu_options, index=0)
        women_edu_mapping = {name: i for i, name in enumerate(women_edu_options)}
        input_data['highest educational level'] = women_edu_mapping[selected_women_edu]

        # Husband/partner education
        partner_edu_options = ["No Education", "Higher", "Primary", "Secondary", "Don't know"]
        selected_partner_edu = st.selectbox("Partner's Education Level", options=partner_edu_options, index=0)
        partner_edu_mapping = {name: i for i, name in enumerate(partner_edu_options)}
        input_data["husband/partner's education level"] = partner_edu_mapping[selected_partner_edu]

        # Women occupation
        women_occ_options = ["Agriculture", "Other", "Professional/Tech/Clerical", "Sales/Service", "Unemployed"]
        selected_women_occ = st.selectbox("Women's Occupation", options=women_occ_options, index=0)
        women_occ_mapping = {name: i for i, name in enumerate(women_occ_options)}
        input_data["respondent's occupation"] = women_occ_mapping[selected_women_occ]

        # Partner occupation
        partner_occ_options = ["Agriculture", "Other", "Professional/Tech/Clerical", "Sales/Service", "Unemployed"]
        selected_partner_occ = st.selectbox("Partner's Occupation", options=partner_occ_options, index=0)
        partner_occ_mapping = {name: i for i, name in enumerate(partner_occ_options)}
        input_data["partner's occupation"] = partner_occ_mapping[selected_partner_occ]

        # Wealth index
        wealth_options = ["Poor", "Middle", "Rich"]
        selected_wealth = st.selectbox("Wealth Index", options=wealth_options, index=0)
        wealth_mapping = {name: i for i, name in enumerate(wealth_options)}
        input_data['wealth index'] = wealth_mapping[selected_wealth]

        # Drinking water
        water_options = ["Improved", "Unimproved"]
        selected_water = st.selectbox("Source of Drinking Water", options=water_options, index=0)
        water_mapping = {name: i for i, name in enumerate(water_options)}
        input_data['source of drinking water'] = water_mapping[selected_water]

        # Toilet facility
        toilet_options = ["Improved", "Unimproved"]
        selected_toilet = st.selectbox("Type of Toilet Facility", options=toilet_options, index=0)
        toilet_mapping = {name: i for i, name in enumerate(toilet_options)}
        input_data['type of toilet facility'] = toilet_mapping[selected_toilet]

        # Marital status
        marital_options = ["Married", "Never in Union", "No Longer Married"]
        selected_marital = st.selectbox("Current Marital Status", options=marital_options, index=0)
        marital_mapping = {name: i for i, name in enumerate(marital_options)}
        input_data['current marital status'] = marital_mapping[selected_marital]

        # Currently pregnant
        pregnant_options = ["No", "Yes"]
        selected_pregnant = st.selectbox("Currently Pregnant", options=pregnant_options, index=0)
        pregnant_mapping = {name: i for i, name in enumerate(pregnant_options)}
        input_data['currently pregnant'] = pregnant_mapping[selected_pregnant]

        # Place of delivery
        delivery_options = ["Health Center", "Home Delivery", "Hospital", "Other"]
        selected_delivery = st.selectbox("Place of Delivery", options=delivery_options, index=0)
        delivery_mapping = {name: i for i, name in enumerate(delivery_options)}
        input_data['place of delivery'] = delivery_mapping[selected_delivery]

        # Knowledge of any method (Yes/No)
        method_options = ["No", "Yes"]
        selected_method = st.selectbox("Knowledge of Any Method", options=method_options, index=0)
        method_mapping = {name: i for i, name in enumerate(method_options)}
        input_data['knowledge of any method'] = method_mapping[selected_method]

        # Living children + current pregnancy (Parity)
        #parity_options = ["0_2", "3_5", "6+"]
        #selected_parity = st.selectbox("Living Children + Current Pregnancy", options=parity_options, index=0)
        #parity_mapping = {name: i for i, name in enumerate(parity_options)}
        #input_data['living children + current pregnancy (grouped)'] = parity_mapping[selected_parity]

        # Body mass index
        bmi_options = ["Normal (18.5–24.99)", "Overweight (≥ 25)", "Underweight (< 18.5)"]
        selected_bmi = st.selectbox("Body Mass Index", options=bmi_options, index=0)
        bmi_mapping = {name: i for i, name in enumerate(bmi_options)}
        input_data['body mass index'] = bmi_mapping[selected_bmi]

        # Had diarrhea recently
        diarrhea_options = ["No", "Yes"]
        selected_diarrhea = st.selectbox("Had Diarrhea Recently", options=diarrhea_options, index=0)
        diarrhea_mapping = {name: i for i, name in enumerate(diarrhea_options)}
        input_data['had diarrhea recently'] = diarrhea_mapping[selected_diarrhea]

        # Smokes cigarettes
        #smoke_options = ["No", "Yes"]
        #selected_smoke = st.selectbox("Smokes Cigarettes", options=smoke_options, index=0)
        #smoke_mapping = {name: i for i, name in enumerate(smoke_options)}
        #input_data['smokes cigarettes'] = smoke_mapping[selected_smoke]

        # Delivery by caesarean section
        csection_options = ["No", "Yes"]
        selected_csection = st.selectbox("Delivery by Caesarean Section", options=csection_options, index=0)
        csection_mapping = {name: i for i, name in enumerate(csection_options)}
        input_data['delivery by caesarean section'] = csection_mapping[selected_csection]

        # Use contraceptive method
        contraceptive_options = ["No", "Yes"]
        selected_contraceptive = st.selectbox("Use Contraceptive Method", options=contraceptive_options, index=0)
        contraceptive_mapping = {name: i for i, name in enumerate(contraceptive_options)}
        input_data['use of contraceptive method'] = contraceptive_mapping[selected_contraceptive]

        # Duration of pregnancy
        pregnancy_options = ["Post term (>9 months)", "Preterm (6-8 months)", "Term (9 months)"]
        selected_pregnancy = st.selectbox("Duration of Pregnancy", options=pregnancy_options, index=2)
        pregnancy_mapping = {name: i for i, name in enumerate(pregnancy_options)}
        input_data['duration of pregnancy'] = pregnancy_mapping[selected_pregnancy]

        # Build input DataFrame
        input_df = pd.DataFrame({col: [val] for col, val in input_data.items()})

        # Ensure column order matches training
        input_final = input_df.reindex(columns=selected_features, fill_value=0)

        # Prediction
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0][1]
        label = le_target.inverse_transform([prediction])[0]

        # Display results
        st.success(f"🩺 Prediction: **{label}**")
        st.info(f"🔢 Probability of being anemic: **{probability:.2%}**")

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
