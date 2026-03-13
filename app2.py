import streamlit as st
import json
import os
import base64
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# TPOT availability check
try:
    from tpot import TPOTClassifier, TPOTRegressor
    tpot_available = True
except ImportError:
    tpot_available = False

# Page configuration
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Consolidated CSS (accessible, high contrast) ----------
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0d47a1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Button styling */
    div.stButton > button {
        background: #1E88E5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background: #1565C0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    /* Form input labels */
    .stTextInput label, .stSelectbox label, .stNumberInput label {
        color: #0d47a1 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ---------- User data storage (simple, for demo) ----------
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register_user(email, password, name):
    users = load_users()
    if email in users:
        return False, "Email already registered."
    users[email] = {"name": name, "password": password}  # Note: plaintext, for demo only
    save_users(users)
    return True, "Registration successful. Please log in."

def authenticate_user(email, password):
    users = load_users()
    if email in users and users[email]["password"] == password:
        return True, users[email]["name"]
    return False, None

# ---------- Session state initialization ----------
if "page" not in st.session_state:
    st.session_state.page = "front"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "data" not in st.session_state:
    st.session_state.data = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "model" not in st.session_state:
    st.session_state.model = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "test_data" not in st.session_state:
    st.session_state.test_data = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "app_page" not in st.session_state:
    st.session_state.app_page = "📊 Data Upload"

def go_to(page):
    st.session_state.page = page

# ---------- Front Page ----------
def front_page():
    st.markdown('<h1 class="main-header">Welcome to No-Code ML Platform</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="text-align: center;">
        <p style="font-size: 1.2rem;">Build machine learning models without writing a single line of code.</p>
        <p>Upload your data, explore, and let AutoML do the rest.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Get Started", use_container_width=True):
            go_to("login")
            st.rerun()

# ---------- Login/Register Page ----------
def login_page():
    st.markdown('<h1 class="main-header">Login / Register</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                success, name = authenticate_user(email, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_name = name
                    go_to("dashboard")
                    st.rerun()
                else:
                    st.error("Invalid email or password")
    
    with tab2:
        with st.form("register_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, msg = register_user(email, password, name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    
    if st.button("← Back to Home"):
        go_to("front")
        st.rerun()

# ---------- Data Upload Page ----------
def upload_page():
    st.markdown('<h2 class="sub-header">📊 Upload Your Dataset</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
            <h4>📁 Supported Format</h4>
            <ul>
                <li>CSV files only (.csv)</li>
                <li>Structured tabular data</li>
                <li>Numerical and categorical variables</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
                st.markdown("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                with st.expander("📊 Basic Info"):
                    st.write("**Shape:**", df.shape)
                    col_types = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Missing': df.isnull().sum(),
                        'Unique': df.nunique()
                    })
                    st.dataframe(col_types, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Notes</h4>
            <ul>
                <li>Ensure data is clean</li>
                <li>Remove sensitive info</li>
                <li>Check for missing values</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            st.markdown("### 🎯 Target Column")
            target_col = st.selectbox(
                "Select the target column:",
                options=st.session_state.data.columns,
                index=len(st.session_state.data.columns)-1
            )
            problem_type = st.selectbox("Problem type:", ["Classification", "Regression"])
            if st.button("Set Target & Continue", type="primary", use_container_width=True):
                st.session_state.target_column = target_col
                st.session_state.problem_type = problem_type
                st.session_state.app_page = "🔍 Exploratory Analysis"
                st.rerun()

# ---------- EDA Page ----------
def eda_page():
    st.markdown('<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📊 Data Upload"
            st.rerun()
        return
    
    df = st.session_state.data
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing", df.isnull().sum().sum())
    col4.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    
    # Data types pie
    st.markdown("### 🏷️ Data Types")
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ['Type', 'Count']
    fig = px.pie(dtype_counts, values='Count', names='Type', title="Data Type Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values
    st.markdown("### ⚠️ Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values,
            'Percentage': (missing.values / len(df)) * 100
        }).sort_values('Percentage', ascending=False)
        fig = px.bar(missing_df, x='Column', y='Percentage', title="Missing Values by Column (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("No missing values!")
    
    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.markdown("### 📊 Numerical Columns")
        selected = st.selectbox("Select a numerical column:", num_cols)
        if selected:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=selected, title=f"Distribution of {selected}", nbins=50)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(df, y=selected, title=f"Boxplot of {selected}")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[selected].describe(), use_container_width=True)
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.markdown("### 📊 Categorical Columns")
        selected = st.selectbox("Select a categorical column:", cat_cols)
        if selected:
            top = df[selected].value_counts().head(20)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(x=top.index, y=top.values, title=f"Top categories in {selected}")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.pie(names=top.index, values=top.values, title=f"Proportion of {selected}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    if len(num_cols) > 1:
        st.markdown("### 🔗 Correlation Matrix")
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    
    # Target analysis
    target = st.session_state.target_column
    if target and target in df.columns:
        st.markdown(f"### 🎯 Target: {target}")
        if df[target].dtype in ['int64', 'float64']:
            fig = px.histogram(df, x=target, title=f"Target Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            counts = df[target].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, title="Class Distribution")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Training Page ----------
def training_page():
    st.markdown('<h2 class="sub-header">🤖 Automated Model Training (TPOT)</h2>', unsafe_allow_html=True)
    
    if not tpot_available:
        st.error("TPOT is not installed. Please install it with: `pip install tpot`")
        st.code("pip install tpot", language="bash")
        return
    
    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("Please upload data and set target column first.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📊 Data Upload"
            st.rerun()
        return
    
    df = st.session_state.data
    target = st.session_state.target_column
    problem = st.session_state.problem_type
    
    st.markdown(f"""
    <div class="card">
        <h4>Configuration</h4>
        <ul>
            <li><strong>Problem:</strong> {problem}</li>
            <li><strong>Target:</strong> {target}</li>
            <li><strong>Dataset:</strong> {df.shape}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Training parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        generations = st.slider("Generations", 5, 50, 10)
    with col2:
        pop_size = st.slider("Population size", 10, 100, 50)
        cv = st.slider("CV folds", 2, 10, 5)
    with col3:
        max_time = st.slider("Max time (minutes)", 1, 60, 10)
        random_state = st.number_input("Random seed", 0, 100, 42)
    
    # Preprocessing options
    pre_cols = st.columns(3)
    with pre_cols[0]:
        handle_missing = st.selectbox("Missing values", ["auto", "drop", "impute"])
    with pre_cols[1]:
        scale = st.checkbox("Scale numerical features", True)
    with pre_cols[2]:
        encode = st.checkbox("Encode categorical features", True)
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Preparing data and running TPOT... This may take a while."):
            try:
                X = df.drop(columns=[target])
                y = df[target]
                
                # Handle missing values
                if handle_missing == "drop":
                    X = X.dropna()
                    y = y.loc[X.index]
                elif handle_missing == "impute":
                    # Simple imputation: fill numeric with mean, categorical with mode
                    for col in X.columns:
                        if X[col].dtype in ['int64', 'float64']:
                            X[col].fillna(X[col].mean(), inplace=True)
                        else:
                            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "missing", inplace=True)
                
                # Encode categoricals
                if encode:
                    cat_cols = X.select_dtypes(include=['object', 'category']).columns
                    for col in cat_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                st.session_state.test_data = {'X_test': X_test, 'y_test': y_test}
                
                # Scale
                if scale:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                # Initialize TPOT
                if problem == "Classification":
                    tpot = TPOTClassifier(
                        generations=generations,
                        population_size=pop_size,
                        cv=cv,
                        random_state=random_state,
                        verbosity=2,
                        n_jobs=-1,
                        max_time_mins=max_time
                    )
                else:
                    tpot = TPOTRegressor(
                        generations=generations,
                        population_size=pop_size,
                        cv=cv,
                        random_state=random_state,
                        verbosity=2,
                        n_jobs=-1,
                        max_time_mins=max_time
                    )
                
                # Fit
                tpot.fit(X_train, y_train)
                st.session_state.model = tpot
                st.session_state.predictions = tpot.predict(X_test)
                st.session_state.training_complete = True
                
                st.success("✅ Training complete!")
                st.balloons()
            except Exception as e:
                st.error(f"Error during training: {e}")

# ---------- Evaluation Page ----------
def evaluation_page():
    st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete or st.session_state.model is None:
        st.warning("Please train a model first.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "🤖 Model Training"
            st.rerun()
        return
    
    model = st.session_state.model
    preds = st.session_state.predictions
    y_test = st.session_state.test_data['y_test']
    problem = st.session_state.problem_type
    
    if problem == "Classification":
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        col4.metric("F1", f"{f1:.4f}")
        
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        st.markdown("### Classification Report")
        report = classification_report(y_test, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    else:
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("MSE", f"{mse:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        col4.metric("R²", f"{r2:.4f}")
        
        st.markdown("### Actual vs Predicted")
        fig = px.scatter(x=y_test, y=preds, labels={'x':'Actual', 'y':'Predicted'})
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Residuals")
        residuals = y_test - preds
        fig = px.scatter(x=preds, y=residuals, labels={'x':'Predicted', 'y':'Residuals'})
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Best Pipeline")
    st.code(model.fitted_pipeline_, language='python')
    
    pipeline_code = model.export()
    st.download_button("📥 Download Pipeline Code", data=pipeline_code,
                       file_name="best_pipeline.py", mime="text/python")

# ---------- Prediction Page ----------
def prediction_page():
    st.markdown('<h2 class="sub-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete or st.session_state.model is None:
        st.warning("Please train a model first.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "🤖 Model Training"
            st.rerun()
        return
    
    model = st.session_state.model
    method = st.radio("Prediction method:", ["📤 Upload new data", "✍️ Manual input", "📊 Use test data"])
    
    if method == "📤 Upload new data":
        st.markdown("### Upload CSV for prediction")
        uploaded = st.file_uploader("Choose a CSV file", type=['csv'], key="pred_upload")
        if uploaded:
            try:
                new_df = pd.read_csv(uploaded)
                # Ensure columns match training features
                feature_cols = st.session_state.data.drop(columns=[st.session_state.target_column]).columns.tolist()
                missing = set(feature_cols) - set(new_df.columns)
                if missing:
                    st.warning(f"Missing columns: {missing}. They will be filled with 0.")
                new_df = new_df.reindex(columns=feature_cols, fill_value=0)
                
                st.dataframe(new_df.head(), use_container_width=True)
                if st.button("Predict", type="primary"):
                    preds = model.predict(new_df)
                    results = new_df.copy()
                    results['Prediction'] = preds
                    st.success(f"Predictions for {len(preds)} samples")
                    st.dataframe(results, use_container_width=True)
                    
                    csv = results.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif method == "✍️ Manual input":
        st.markdown("### Enter values")
        feature_cols = st.session_state.data.drop(columns=[st.session_state.target_column]).columns.tolist()
        input_data = {}
        cols = st.columns(3)
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                if st.session_state.data[col].dtype in ['int64', 'float64']:
                    min_val = float(st.session_state.data[col].min())
                    max_val = float(st.session_state.data[col].max())
                    mean_val = float(st.session_state.data[col].mean())
                    input_data[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)
                else:
                    unique = st.session_state.data[col].dropna().unique()[:10]
                    input_data[col] = st.selectbox(col, unique)
        
        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            st.markdown(f"""
            <div class="success-box">
                <h3>Predicted {st.session_state.target_column}: {pred}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # Use test data
        st.markdown("### Test set predictions")
        if st.session_state.test_data:
            X_test = st.session_state.test_data['X_test']
            y_test = st.session_state.test_data['y_test']
            preds = model.predict(X_test)
            comp = X_test.copy()
            comp['Actual'] = y_test.values
            comp['Predicted'] = preds
            st.dataframe(comp.head(20), use_container_width=True)
            
            csv = comp.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="test_predictions.csv">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No test data available.")

# ---------- Export Page ----------
def export_page():
    st.markdown('<h2 class="sub-header">💾 Export Model & Results</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete:
        st.warning("Please train a model first.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "🤖 Model Training"
            st.rerun()
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🐍 Python Pipeline")
        if st.button("Generate pipeline code"):
            code = st.session_state.model.export()
            st.code(code, language='python')
            st.download_button("📥 Download", data=code, file_name="tpot_pipeline.py", mime="text/python")
    
    with col2:
        st.markdown("#### 📊 Model Report")
        if st.button("Generate report"):
            report = f"""# Machine Learning Model Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Problem Type:** {st.session_state.problem_type}
**Target Column:** {st.session_state.target_column}
**Dataset Shape:** {st.session_state.data.shape if st.session_state.data is not None else 'N/A'}

**Best Pipeline:**
{st.session_state.model.fitted_pipeline_}

**Training Complete:** {st.session_state.training_complete}
"""
            st.code(report, language='markdown')
            st.download_button("📥 Download", data=report, file_name="model_report.md", mime="text/markdown")
    
    st.markdown("### 📋 Session Info")
    info = {
        "Data loaded": st.session_state.data is not None,
        "Target set": st.session_state.target_column is not None,
        "Model trained": st.session_state.training_complete,
        "Predictions available": st.session_state.predictions is not None
    }
    st.dataframe(pd.DataFrame.from_dict(info, orient='index', columns=['Status']), use_container_width=True)
    
    st.markdown("### 🔄 Reset Session")
    if st.button("Reset all data", type="secondary"):
        keys = ["data", "target_column", "problem_type", "model", "predictions", "test_data", "training_complete"]
        for k in keys:
            if k in st.session_state:
                st.session_state[k] = None
        st.rerun()

# ---------- Dashboard ----------
def dashboard_page():
    st.markdown(f"<h1 style='color: #0d47a1;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=80)
        st.markdown("### Navigation")
        pages = [
            "📊 Data Upload",
            "🔍 Exploratory Analysis",
            "🤖 Model Training",
            "📈 Model Evaluation",
            "🔮 Make Predictions",
            "💾 Export Results"
        ]
        selected = st.radio("Go to", pages, index=pages.index(st.session_state.app_page))
        st.session_state.app_page = selected
        
        st.markdown("---")
        st.info("**Info**\n- Upload CSV\n- Auto EDA\n- TPOT AutoML\n- No-code predictions")
        if not tpot_available:
            st.error("TPOT not installed")
        
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            for k in ["data", "target_column", "problem_type", "model", "predictions", "test_data", "training_complete"]:
                if k in st.session_state:
                    st.session_state[k] = None
            go_to("front")
            st.rerun()
    
    # Main content
    if st.session_state.app_page == "📊 Data Upload":
        upload_page()
    elif st.session_state.app_page == "🔍 Exploratory Analysis":
        eda_page()
    elif st.session_state.app_page == "🤖 Model Training":
        training_page()
    elif st.session_state.app_page == "📈 Model Evaluation":
        evaluation_page()
    elif st.session_state.app_page == "🔮 Make Predictions":
        prediction_page()
    elif st.session_state.app_page == "💾 Export Results":
        export_page()

# ---------- Main Router ----------
if st.session_state.page == "front":
    front_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "dashboard":
    if not st.session_state.logged_in:
        go_to("login")
        st.rerun()
    else:
        dashboard_page()