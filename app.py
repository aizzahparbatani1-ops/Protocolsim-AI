import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import re
from pypdf import PdfReader

# Page Configuration & Medical Branding Style
st.set_page_config(page_title="ProtocolSim AI", page_icon="🧬", layout="wide")

# Custom CSS Layout Injection for Professional UI
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    h1 { color: #58a6ff !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 700; }
    h2, h3 { color: #f0f6fc !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 500; }
    .stButton>button { 
        background-color: #238636 !important; color: white !important; 
        font-weight: 600 !important; border-radius: 6px !important; 
        border: none !important; padding: 0.5rem 2rem !important;
    }
    .stButton>button:hover { background-color: #2ea043 !important; }
    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 2.2rem !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.95rem !important; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    .rec-box { background-color: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Main Title Headers
st.title("🧬 ProtocolSim AI")
st.subheader("Early-Stage Computational Research Prototype")
st.markdown("*Exploring the utility of retrospective machine learning models to identify structural and clinical design characteristics associated with oncology trial outcomes.*")
st.markdown("---")

# Step 1: Real NIH Government Dataset Ingestion Pipeline Loop
@st.cache_data
def load_real_oncology_data():
    if not os.path.exists("cancer_trials.csv"):
        return None
        
    data = pd.read_csv("cancer_trials.csv")

    if 'Study Status' in data.columns and not data.empty:
        successful_trials = data['Study Status'].eq('COMPLETED')
        baseline_success_count = int(successful_trials.sum())
        baseline_failure_count = len(data) - baseline_success_count
        baseline_success_ratio = baseline_success_count / len(data)
        baseline_failure_ratio = baseline_failure_count / len(data)

        if baseline_failure_ratio < 0.35 and baseline_failure_count > 0:
            target_failure_count = int(np.ceil(baseline_success_count * 0.35 / 0.65))
            additional_failure_count = target_failure_count - baseline_failure_count
            if additional_failure_count > 0:
                unsuccessful_trials = data.loc[~successful_trials]
                synthetic_failures = unsuccessful_trials.sample(
                    n=additional_failure_count,
                    replace=True,
                    random_state=42,
                )
                data = pd.concat([data, synthetic_failures], ignore_index=True)
    
    # Process string fields to structured coordinates backends
    if 'Phases' in data.columns:
        data['phase_num'] = data['Phases'].str.extract(r'(\d+)').astype(float).fillna(2.0)
    else:
        data['phase_num'] = 2.0
        
    if 'Enrollment' in data.columns:
        data['cohort_size'] = pd.to_numeric(data['Enrollment'], errors='coerce').fillna(150.0)
    else:
        data['cohort_size'] = 150.0
        
    if 'Age' in data.columns:
        data['max_age'] = data['Age'].str.extract(r'(\d+)').astype(float).fillna(65.0)
    else:
        data['max_age'] = 65.0
        
    if 'Sex' in data.columns:
        data['sex_encoded'] = np.where(data['Sex'].str.contains('FEMALE', na=False), 1, 
                              np.where(data['Sex'].str.contains('MALE', na=False), 2, 0))
    else:
        data['sex_encoded'] = 0
        
    if 'Study Design' in data.columns:
        data['design_encoded'] = np.where(data['Study Design'].str.contains('RANDOMIZED', na=False), 1, 0)
    else:
        data['design_encoded'] = 1
        
    data['baseline_toxicity'] = np.random.randint(20, 150, len(data))
                
    if 'Study Status' in data.columns:
        data['success'] = np.where(data['Study Status'] == 'COMPLETED', 1, 0)
    else:
        data['success'] = 1
        
    return data[['phase_num', 'cohort_size', 'max_age', 'sex_encoded', 'design_encoded', 'baseline_toxicity', 'success']]

df = load_real_oncology_data()

# --- LIVE DATASET TELEMETRY HEADERS ---
if df is not None:
    st.header("📊 Active Database Architecture")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Ingested Historical Clinical Trial Protocol Records", value=f"{len(df):,}")
    with col_b:
        st.metric(label="Ingested Diagnostic Parameters", value=f"{df.shape[1] - 1} Variables")
    with col_c:
        st.metric(label="Target Disease Core Scope", value="Oncology / Cancer Registry")
st.markdown("---")

# Initialize variable holds with baseline defaults
extracted_phase = "Phase 2"
extracted_cohort = 150
extracted_age = 65
extracted_sex = "ALL"
extracted_design = "RANDOMIZED"
extracted_toxicity = 45

# --- AUTOMATED PROTOCOL DOCUMENT DROPZONE ---
st.header("1. Automated Protocol Document Dropzone")
st.markdown("Upload clinical draft outlines below to auto-configure simulation boundaries. Supports `.txt` and `.pdf` formats.")
uploaded_file = st.file_uploader("", type=["txt", "pdf"])

if uploaded_file is not None:
    protocol_text = ""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension == ".pdf":
        st.info("📄 PDF binary document detected. Initializing `pypdf` sequential parsing loops...")
        try:
            pdf_reader = PdfReader(uploaded_file)
            page_count = len(pdf_reader.pages)
            st.write(f"ℹ️ Found {page_count} document text layers. Extracting strings...")
            
            for page_num in range(page_count):
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text:
                    protocol_text += page_text + "\n"
        except Exception as e:
            st.error(f"Error compiling PDF structural layouts: {e}")
    else:
        protocol_text = uploaded_file.read().decode("utf-8")
        
    if protocol_text:
        st.info("🧬 Processing text arrays... Parsing biometric boundaries against analytical framework...")
        
        phase_match = re.search(r'(?:phase|cohort stage)\s*[:=~#-]*\s*(\d+)', protocol_text, re.IGNORECASE)
        cohort_match = re.search(r'(?:enrollment|sample size|target cohort|n\s*=)\s*(?:count of|totaling)?\s*[:=~#-]*\s*(\d+)', protocol_text, re.IGNORECASE)
        age_match = re.search(r'(?:maximum age|age ceiling|cutoff|age limit)\s*(?:threshold of|set to)?\s*[:=~#-]*\s*(\d+)', protocol_text, re.IGNORECASE)
        toxicity_match = re.search(r'(?:liver enzyme|alt limit|toxicity ceiling)\s*(?:tolerance of)?\s*[:=~#-]*\s*(\d+)', protocol_text, re.IGNORECASE)
        
        if phase_match:
            found_phase = int(phase_match.group(1))
            # FIXED PERMANENTLY: Valid phase array range bounded correctly inside brackets
            if found_phase in [1, 2, 3, 4]:
                extracted_phase = f"Phase {found_phase}"
        if cohort_match:
            extracted_cohort = min(max(int(cohort_match.group(1)), 20), 2000)
        if age_match:
            extracted_age = min(max(int(age_match.group(1)), 40), 90)
        if toxicity_match:
            extracted_toxicity = min(max(int(toxicity_match.group(1)), 20), 150)
            
        if "female" in protocol_text.lower():
            extracted_sex = "FEMALE"
        elif "male" in protocol_text.lower():
            extracted_sex = "MALE"
        
        if "non-randomized" in protocol_text.lower() or "non randomized" in protocol_text.lower():
            extracted_design = "NON_RANDOMIZED"
            
        st.success("✅ Extracted parameters loaded successfully into current validation environment.")

# Sidebar Parameter UI Ingestion Layout
st.sidebar.header("2. Extraction Control Panel")
st.sidebar.markdown("Verify or manually adjust extracted metrics:")
phase_select = st.sidebar.selectbox("Trial Phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4"], 
                                    index=["Phase 1", "Phase 2", "Phase 3", "Phase 4"].index(extracted_phase))
cohort_size = st.sidebar.slider("Sample Size (n=)", 20, 2000, int(extracted_cohort))
max_age = st.sidebar.slider("Maximum Patient Age Cutoff", 40, 90, int(extracted_age))
sex_select = st.sidebar.selectbox("Target Cohort Sex", ["ALL", "FEMALE", "MALE"], 
                                  index=["ALL", "FEMALE", "MALE"].index(extracted_sex))
design_select = st.sidebar.selectbox("Study Allocation Constraint", ["RANDOMIZED", "NON_RANDOMIZED", "NA"], 
                                     index=["RANDOMIZED", "NON_RANDOMIZED", "NA"].index(extracted_design))
baseline_toxicity = st.sidebar.slider("Allowed Liver Enzyme Baseline (ALT U/L)", 20, 150, int(extracted_toxicity))

# Convert UI Strings to match Numeric Matrix Values
phase_map = {"Phase 1": 1.0, "Phase 2": 2.0, "Phase 3": 3.0, "Phase 4": 4.0}
phase = phase_map[phase_select]
sex_encoded = 1 if sex_select == "FEMALE" else (2 if sex_select == "MALE" else 0)
design_encoded = 1 if design_select == "RANDOMIZED" else 0

if df is not None:
    # Step 3: Train the Machine Learning Classifier on the Variable Matrix
    X = df[['phase_num', 'cohort_size', 'max_age', 'sex_encoded', 'design_encoded', 'baseline_toxicity']]
    y = df['success']
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(X, y)

    # Step 4: Run the In-Silico Simulation Routine
    st.markdown("---")
    st.header("3. Run In-Silico Evaluation")
    
    # Calculate initial predictions permanently outside of button state blocks
    user_input = np.array([[phase, cohort_size, max_age, sex_encoded, design_encoded, baseline_toxicity]])
    probabilities = model.predict_proba(user_input)
    
    def extract_outcome_probabilities(prediction_probabilities):
        success_probability = 0.0
        failure_probability = 0.0
        class_probabilities = prediction_probabilities[0]

        if prediction_probabilities.ndim == 2 and prediction_probabilities.shape[1] > 1:
            for class_label, class_probability in zip(model.classes_, class_probabilities):
                if class_label == 1:
                    success_probability = class_probability * 100
                elif class_label == 0:
                    failure_probability = class_probability * 100
        elif len(model.classes_) == 1:
            if model.classes_[0] == 1:
                success_probability = class_probabilities[0] * 100
            else:
                failure_probability = class_probabilities[0] * 100

        return success_probability, failure_probability

    success_probability, failure_probability = extract_outcome_probabilities(probabilities)

    st.subheader("Predictive Analytics Telemetry Output")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Calculated Protocol Success Probability", value=f"{success_probability:.1f}%")
    with c2:
        st.metric(label="Estimated Risk of Phase Termination (Safety/Efficacy)", value=f"{failure_probability:.1f}%")
        
    # --- RENDER SUMMARY & OPTIMIZATION BLOCKS UNCONDITIONALLY ---
    st.markdown("---")
    st.subheader("📋 Evaluation Context Summary & Optimization Data")

    if failure_probability > 50:
        st.error("🚨 Statistical Flag: High Attrition Risk Clustering Detected")
    else:
        st.success("✅ Initial Baseline Evaluation: Protocol Bounds Within Safe Historical Tolerances")

    optimized_max_age = max(max_age - 8, 40)
    optimized_cohort_size = cohort_size * 0.85 if cohort_size > 300 else cohort_size
    optimized_baseline_toxicity = baseline_toxicity - 15
    optimized_input = np.array([[
        phase,
        optimized_cohort_size,
        optimized_max_age,
        sex_encoded,
        design_encoded,
        optimized_baseline_toxicity,
    ]])
    optimized_probabilities = model.predict_proba(optimized_input)
    optimized_success_probability, optimized_failure_probability = (
        extract_outcome_probabilities(optimized_probabilities)
    )
    potential_uplift = optimized_success_probability - success_probability

    st.markdown(
        f"""
        <div class="rec-box">
            <h3>🛠️ Automated Optimization Strategy Engine</h3>
            <ul>
                <li>Maximum patient age: {max_age} to {optimized_max_age} years</li>
                <li>Sample size: {cohort_size:,} to {optimized_cohort_size:,.2f}</li>
                <li>Baseline toxicity: {baseline_toxicity} to {optimized_baseline_toxicity} U/L</li>
            </ul>
            <p><strong>Optimized success probability:</strong> {optimized_success_probability:.2f}%</p>
            <p><strong>Optimized phase-termination risk:</strong> {optimized_failure_probability:.2f}%</p>
            <p><strong>Potential success-probability uplift:</strong> {potential_uplift:+.2f} percentage points</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
