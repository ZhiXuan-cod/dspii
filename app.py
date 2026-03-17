import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

from supabase import create_client

# ---------- PyCaret availability ----------
try:
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare, predict_model as clf_predict, pull as clf_pull, save_model, load_model, plot_model
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, predict_model as reg_predict, pull as reg_pull
    pycaret_available = True
except ImportError:
    pycaret_available = False

# Try to import get_config (internal PyCaret helper) – may fail in some versions
try:
    from pycaret.internal.pycaret_experiment import get_config
    get_config_available = True
except ImportError:
    get_config_available = False

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Supabase 客户端 ----------
if "supabase" not in st.session_state:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        st.session_state.supabase = create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        st.session_state.supabase = None

# ---------- 背景图片（Base64嵌入）----------
def get_base64_of_file(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_bg_image_local(image_path):
    bin_str = get_base64_of_file(image_path)
    if bin_str:
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        fallback_bg = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """
        st.markdown(fallback_bg, unsafe_allow_html=True)

# ---------- 用户数据存储 (Supabase) ----------
def register_user(email, password, name):
    """注册新用户到 Supabase"""
    if st.session_state.supabase is None:
        return False, "Supabase not connected"
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) > 0:
            return False, "Email already registered."
        data = {"email": email, "name": name, "password": password}
        st.session_state.supabase.table("users").insert(data).execute()
        return True, "Registration successful. Please log in."
    except Exception as e:
        return False, f"Registration failed: {e}"

def authenticate_user(email, password):
    """验证用户凭据"""
    if st.session_state.supabase is None:
        return False, None
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) == 0:
            return False, None
        user = response.data[0]
        if user["password"] == password:  # 建议生产环境使用密码哈希
            return True, user["name"]
        else:
            return False, None
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False, None

# ---------- 页面导航 ----------
if "page" not in st.session_state:
    st.session_state.page = "front"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

def go_to(page):
    st.session_state.page = page

# ---------- 全局CSS样式（按钮渐变、卡片等）----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3949AB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: rgba(255,255,255,0.85);
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
    /* 全局按钮样式：蓝到紫渐变 */
    div.stButton > button {
        background: linear-gradient(135deg, #2196F3, #9C27B0) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.2rem !important;
        border-radius: 50px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
    div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 首页 Front Page ----------
def front_page():
    set_bg_image_local("FrontPage.jpg")

    st.markdown("""
    <style>
    .right-panel {
        background-color: rgba(0, 0, 0, 0.70);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        animation: fadeIn 1s ease-in-out;
        color: white;
    }
    .right-panel h1 {
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .right-panel p {
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-size: 1.2rem;
        opacity: 0.9;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .st-emotion-cache-ocqkz7 {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        video_path = "animation.mp4"
        if os.path.exists(video_path):
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode()
            video_html = f"""
            <video width="100%" autoplay loop muted playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 2rem; text-align: center;">
                <span style="font-size: 3rem;">📹</span>
                <p style="color: white;">Video not found. Please add animation.mp4</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        right_html = """
        <div class="right-panel">
            <h1>Welcome to<br>No-Code ML Platform</h1>
            <p>Accessible for Machine Learning without code.</p>
        </div>
        """
        st.markdown(right_html, unsafe_allow_html=True)
        if st.button("Get Started", key="get_started", use_container_width=True):
            go_to("login")
            st.rerun()

# ---------- 登录/注册页面（居中）----------
def login_page():
    set_bg_image_local("login.jpg")
    
    st.markdown("""
    <style>
    /* 使内容垂直居中 */
    .stApp {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: white;
        border-bottom-color: #2196F3;
    }
    .stTextInput input {
        color: black !important;
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 5px;
    }
    .stTextInput label {
        color: black !important;
    }
    .back-button-container {
        text-align: center;
        margin-top: 1.5rem;
    }
    .back-button-container button {
        background: transparent !important;
        color: rgba(255,255,255,0.9) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 50px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        width: auto !important;
        display: inline-block !important;
        box-shadow: none !important;
    }
    .back-button-container button:hover {
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.6) !important;
        transform: scale(1.02) !important;
    }
    .back-button-container button:active {
        transform: scale(0.98) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown("<h2 style='color: white; text-align: center; margin-bottom: 1.5rem;'>Login / Register</h2>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")  
                password = st.text_input("Password", type="password")  
                submitted = st.form_submit_button("Login")
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
                submitted = st.form_submit_button("Register")
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
        
        st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
        if st.button("← Back to Home", key="back_home"):
            go_to("front")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- 初始化会话状态 ----------
if "data" not in st.session_state:
    st.session_state.data = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "model" not in st.session_state:
    st.session_state.model = None          # 存储 PyCaret 训练好的模型
if "experiment" not in st.session_state:
    st.session_state.experiment = None     # 存储 PyCaret 实验设置
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "test_data" not in st.session_state:
    st.session_state.test_data = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "app_page" not in st.session_state:
    st.session_state.app_page = "📁 Data Upload"

# ---------- 以下为Dashboard子页面 ----------
def upload_page():
    st.markdown('<h2 class="sub-header">📁 Upload Your Dataset</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
        <h4>📁 Supported Data Format</h4>
        <ul>
            <li>CSV files only (.csv)</li>
            <li>Structured tabular data</li>
            <li>Numerical and categorical variables</li>
            <li>Clear target column for supervised learning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"✔️ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                st.markdown("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                with st.expander("📊 Basic Data Statistics"):
                    st.write("**Shape:**", df.shape)
                    col_types = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Missing Values': df.isnull().sum(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(col_types, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Important Notes</h4>
        <ul>
            <li>Ensure your data is clean</li>
            <li>Remove sensitive information</li>
            <li>Check for missing values</li>
            <li>Define target variable clearly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.data is not None:
            st.markdown("### 📌 Define Target Column")
            target_col = st.selectbox(
                "Select the target column:",
                options=st.session_state.data.columns.tolist(),
                index=len(st.session_state.data.columns)-1
            )
            problem_type = st.selectbox("Select problem type:", ["Classification", "Regression"])
            if st.button("Set Target & Continue", type="primary"):
                st.session_state.target_column = target_col
                st.session_state.problem_type = problem_type
                st.success(f"✅ Target set: {target_col} ({problem_type})")
                st.session_state.app_page = "🔍 Exploratory Analysis"
                st.rerun()

def eda_page():
    st.markdown('<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first from the 'Data Upload' page.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()
        return
    df = st.session_state.data

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Rows", len(df))
    with info_col2:
        st.metric("Columns", len(df.columns))
    with info_col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    with info_col4:
        memory = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory (MB)", f"{memory:.2f}")

    st.markdown("### 🔍 Data Types")
    dtype_counts = df.dtypes.value_counts()
    dtype_df = pd.DataFrame({
        'Data Type': dtype_counts.index.astype(str),
        'Count': dtype_counts.values
    })
    fig = px.pie(dtype_df, values='Count', names='Data Type', title="Distribution of Data Types")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚠️ Missing Values Analysis")
    missing_series = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_series.index,
        'Missing_Count': missing_series.values,
        'Missing_Percentage': (missing_series.values / len(df)) * 100
    }).sort_values('Missing_Percentage', ascending=False)
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    if len(missing_df) > 0:
        fig = px.bar(missing_df, x='Column', y='Missing_Percentage',
                    title="Missing Values by Column (%)", color='Missing_Percentage')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values found!")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        st.markdown("### 📊 Numerical Columns Analysis")
        selected_num_col = st.selectbox("Select numerical column:", numerical_cols)
        if selected_num_col:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}", nbins=50)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
                st.plotly_chart(fig, use_container_width=True)
            stats = df[selected_num_col].describe()
            st.dataframe(stats, use_container_width=True)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        st.markdown("### 📊 Categorical Columns Analysis")
        selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
        if selected_cat_col:
            value_counts = df[selected_cat_col].value_counts().head(20)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                            title=f"Top Categories in {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(names=value_counts.index, values=value_counts.values,
                            title=f"Distribution of {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)

    if len(numerical_cols) > 1:
        st.markdown("### 🔗 Correlation Matrix")
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        title="Correlation Heatmap")
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.target_column and st.session_state.target_column in df.columns:
        st.markdown(f"### 🎯 Analysis of Target: {st.session_state.target_column}")
        target_col = st.session_state.target_column
        if df[target_col].dtype in ['int64', 'float64']:
            fig = px.histogram(df, x=target_col, title=f"Distribution of Target ({target_col})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            value_counts = df[target_col].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(x=value_counts.index, y=value_counts.values, title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(names=value_counts.index, values=value_counts.values, title="Class Proportions")
                st.plotly_chart(fig, use_container_width=True)

# ========== 模型名称映射 ==========
CLASSIFICATION_MODEL_MAP = {
    "Logistic Regression": "lr",
    "Random Forest": "rf",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Decision Tree": "dt",
    "Ridge": "ridge",
    "KNN": "knn",
    "SVM": "svm",
    "Naive Bayes": "nb"
}

REGRESSION_MODEL_MAP = {
    "Linear Regression": "lr",
    "Random Forest": "rf",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Decision Tree": "dt",
    "Ridge": "ridge",
    "Lasso": "lasso",
    "KNN": "knn",
    "SVM": "svm"
}

# ========== 增强数据清洗的训练页面 ==========
def training_page():
    st.markdown('<h2 class="sub-header">📐 Automated Model Training with PyCaret</h2>', unsafe_allow_html=True)
    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload data and set target column first.")
        if st.button("Go to Data Upload"):
            st.session_state.app_page = "📁 Data Upload"
            st.rerun()
        return
    if not pycaret_available:
        st.error("PyCaret is not installed. Please install it to use AutoML features.")
        st.code("pip install pycaret", language="bash")
        return

    df = st.session_state.data
    target_col = st.session_state.target_column
    problem_type = st.session_state.problem_type

    st.markdown(f"""
    <div class="card">
    <h4>Training Configuration</h4>
    <ul>
        <li><strong>Problem Type:</strong> {problem_type}</li>
        <li><strong>Target Column:</strong> {target_col}</li>
        <li><strong>Dataset Shape:</strong> {df.shape}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # 获取数值列和分类列（用于后续选项）
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 初始化会话状态变量（可选，但保持整洁）
    if "train_folds" not in st.session_state:
        st.session_state.train_folds = 5
    if "train_metric" not in st.session_state:
        st.session_state.train_metric = "Accuracy" if problem_type == "Classification" else "R2"
    if "train_tune" not in st.session_state:
        st.session_state.train_tune = False
    if "train_tune_iters" not in st.session_state:
        st.session_state.train_tune_iters = 10

    # 参数配置区域：使用两列布局，左侧预处理，右侧模型选择
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧹 Data Preprocessing")
        
        # 缺失值插补
        numeric_imputation = st.selectbox(
            "Numeric missing value imputation",
            ["mean", "median", "mode", "zero"],
            help="Strategy for imputing missing values in numeric columns"
        )
        categorical_imputation = st.selectbox(
            "Categorical missing value imputation",
            ["mode", "constant"],
            help="Strategy for imputing missing values in categorical columns"
        )
        
        # 异常值处理
        remove_outliers = st.checkbox("Remove outliers", value=False,
                                      help="Detect and remove outliers using PCA linear approximation")
        outliers_threshold = None
        if remove_outliers:
            outliers_threshold = st.slider("Outliers threshold", 0.01, 0.5, 0.05, step=0.01,
                                          help="Fraction of data to consider as outliers")
        
        # 特征分箱（仅对数值列）
        bin_numeric_features = []
        if numerical_cols:
            bin_numeric_features = st.multiselect(
                "Bin numeric features (convert to categorical)",
                numerical_cols,
                help="Selected numeric columns will be discretized into quantile bins"
            )
        
        # 组合稀有类别
        combine_rare_levels = st.checkbox("Combine rare levels in categorical features", value=False,
                                          help="Group rare categories into a single 'rare' level")
        rare_level_threshold = None
        if combine_rare_levels and categorical_cols:
            rare_level_threshold = st.slider("Rare level threshold", 0.01, 0.2, 0.05, step=0.01,
                                             help="Minimum percentage to keep a category; below this threshold will be combined")
        
        # 特征选择
        feature_selection = st.checkbox("Use feature selection", value=False,
                                        help="Select a subset of features based on importance")
        feature_selection_threshold = None
        if feature_selection:
            feature_selection_threshold = st.slider("Feature selection threshold", 0.5, 1.0, 0.8, step=0.05,
                                                    help="Percentile of features to keep (based on importance)")
        
        # 其他常用预处理
        normalize = st.checkbox("Normalize numerical features", value=False)
        transformation = st.checkbox("Apply power transformation", value=False)
        remove_multicollinearity = st.checkbox("Remove multicollinearity", value=False)
        ignore_low_variance = st.checkbox("Ignore low variance features", value=False)
        polynomial_features = st.checkbox("Create polynomial features", value=False)
        
        # 数据拆分比例
        train_size = st.slider("Training data fraction", 0.6, 0.9, 0.8, step=0.05)

    with col2:
        st.markdown("#### 🎯 Model Selection")
        
        # 模型选择
        if problem_type == "Classification":
            model_display_names = list(CLASSIFICATION_MODEL_MAP.keys())
        else:
            model_display_names = list(REGRESSION_MODEL_MAP.keys())
        selected_displays = st.multiselect("Include models (leave empty for all)", model_display_names, default=[])
        if not selected_displays:
            include_models = None
        else:
            model_map = CLASSIFICATION_MODEL_MAP if problem_type == "Classification" else REGRESSION_MODEL_MAP
            include_models = [model_map[name] for name in selected_displays if name in model_map]

        # 交叉验证折数
        folds = st.slider("Cross-validation folds", 2, 10, value=st.session_state.train_folds, key="train_folds")

        # 评估指标
        if problem_type == "Classification":
            metric_options = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
        else:
            metric_options = ["R2", "MAE", "MSE", "RMSE", "RMSLE", "MAPE"]
        metric = st.selectbox("Optimization metric", metric_options,
                              index=metric_options.index(st.session_state.train_metric) if st.session_state.train_metric in metric_options else 0,
                              key="train_metric")

        # 超参数调优
        tune = st.checkbox("Tune hyperparameters of the best model", value=st.session_state.train_tune, key="train_tune")
        if tune:
            tune_iters = st.slider("Tuning iterations", 5, 50, value=st.session_state.train_tune_iters, key="train_tune_iters")

    st.warning("⚠️ Streamlit Cloud免费环境内存有限（约1GB）。若数据集较大或选择复杂模型，训练可能失败。建议使用较小的数据样本或简化预处理。")

    if st.button("🚀 Start Automated Training", type="primary", use_container_width=True):
        with st.spinner("🧠 PyCaret is setting up the environment and training models. This may take several minutes..."):
            try:
                # 根据问题类型选择相应的 setup 和 compare 函数
                if problem_type == "Classification":
                    setup_func = clf_setup
                    compare_func = clf_compare
                    pull_func = clf_pull
                else:
                    setup_func = reg_setup
                    compare_func = reg_compare
                    pull_func = reg_pull

                # 构建 setup 参数（只传递用户启用的选项）
                setup_params = {
                    "data": df,
                    "target": target_col,
                    "train_size": train_size,
                    "normalize": normalize,
                    "transformation": transformation,
                    "remove_multicollinearity": remove_multicollinearity,
                    "ignore_low_variance": ignore_low_variance,
                    "polynomial_features": polynomial_features,
                    "numeric_imputation": numeric_imputation,
                    "categorical_imputation": categorical_imputation,
                    "fold_strategy": 'kfold',
                    "n_jobs": 1,  # 限制线程避免内存爆炸
                    "session_id": 42,
                    "verbose": False,
                    "silent": True
                }

                # 添加可选参数（仅当用户启用时）
                if remove_outliers:
                    setup_params["remove_outliers"] = True
                    if outliers_threshold is not None:
                        setup_params["outliers_threshold"] = outliers_threshold
                if bin_numeric_features:
                    setup_params["bin_numeric_features"] = bin_numeric_features
                if combine_rare_levels:
                    setup_params["combine_rare_levels"] = True
                    if rare_level_threshold is not None:
                        setup_params["rare_level_threshold"] = rare_level_threshold
                if feature_selection:
                    setup_params["feature_selection"] = True
                    if feature_selection_threshold is not None:
                        setup_params["feature_selection_threshold"] = feature_selection_threshold

                # 调用 PyCaret setup
                exp = setup_func(**setup_params)
                st.session_state.experiment = exp

                st.info("PyCaret setup completed. Preprocessing applied successfully.")

                # 比较模型
                with st.spinner("Comparing models..."):
                    best_model = compare_func(
                        include=include_models,
                        fold=folds,
                        sort=metric,
                        n_select=1,
                        verbose=False
                    )
                st.session_state.model = best_model
                st.session_state.training_complete = True

                # 尝试获取测试数据
                if get_config_available:
                    try:
                        X_test = get_config('X_test')
                        y_test = get_config('y_test')
                        st.session_state.test_data = {'X_test': X_test, 'y_test': y_test}
                    except Exception as e:
                        st.warning(f"Could not retrieve test data from experiment: {e}")
                        st.session_state.test_data = None
                else:
                    st.info("Test data storage not available – evaluation will use the model's internal predictions.")
                    st.session_state.test_data = None

                # 对测试集进行预测并存储（如果有测试数据）
                if st.session_state.test_data is not None:
                    predictions = predict_model(best_model, data=st.session_state.test_data['X_test'])
                    st.session_state.predictions = predictions['prediction_label'] if problem_type == "Classification" else predictions['prediction_label']

                st.success("🎉 Model training completed successfully!")
                st.balloons()

                # 显示模型比较结果
                results = pull_func()
                st.markdown("### 📊 Model Comparison Results")
                st.dataframe(results, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.exception(e)

# ---------- 评估页面 (保持不变) ----------
def evaluation_page():
    st.markdown('<h2 class="sub-header">📈 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    if not st.session_state.training_complete or st.session_state.model is None:
        st.warning("⚠️ Please train a model first from the 'Model Training' page.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "📐 Model Training"
            st.rerun()
        return

    model = st.session_state.model
    problem_type = st.session_state.problem_type
    test_data = st.session_state.test_data

    if test_data is None:
        st.warning("Test data not available. The model was trained but evaluation cannot be shown. You can still make predictions.")
        st.markdown("### 🏆 Best Model Details")
        st.write(model)
        return

    X_test = test_data['X_test']
    y_test = test_data['y_test']

    if problem_type == "Classification":
        predictions = predict_model(model, data=X_test)
        y_pred = predictions['prediction_label']
    else:
        predictions = predict_model(model, data=X_test)
        y_pred = predictions['prediction_label']

    if problem_type == "Classification":
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")

        st.markdown("### 🎯 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        st.markdown("### 📝 Detailed Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        st.markdown("### 📊 Additional Plots (PyCaret)")
        plot_types = ['auc', 'confusion_matrix', 'class_report', 'feature', 'learning']
        selected_plot = st.selectbox("Select plot type", plot_types)
        if st.button("Generate Plot"):
            try:
                plot_model(model, plot=selected_plot, display_format='streamlit')
            except Exception as e:
                st.error(f"Could not generate plot: {e}")

    else:  # Regression
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("MSE", f"{mse:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        col4.metric("R² Score", f"{r2:.4f}")

        st.markdown("### 📈 Actual vs Predicted")
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
            title='Actual vs Predicted Values')
        max_val = max(max(y_test), max(y_pred))
        min_val = min(min(y_test), min(y_pred))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📉 Residual Plot")
        residuals = y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'},
                        title='Residual Plot')
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Additional Plots (PyCaret)")
        plot_types = ['residuals', 'error', 'cooks', 'learning', 'feature']
        selected_plot = st.selectbox("Select plot type", plot_types)
        if st.button("Generate Plot"):
            try:
                plot_model(model, plot=selected_plot, display_format='streamlit')
            except Exception as e:
                st.error(f"Could not generate plot: {e}")

    st.markdown("### 🏆 Best Model Details")
    st.write(model)

# ---------- 预测页面 (保持不变) ----------
def prediction_page():
    st.markdown('<h2 class="sub-header">🔮 Make Predictions with Trained Model</h2>', unsafe_allow_html=True)
    if not st.session_state.training_complete or st.session_state.model is None:
        st.warning("⚠️ Please train a model first from the 'Model Training' page.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "📐 Model Training"
            st.rerun()
        return

    model = st.session_state.model
    problem_type = st.session_state.problem_type

    method = st.radio("Select prediction method:",
                    ["📤 Upload New Data", "✍️ Manual Input", "📊 Use Test Data"])

    if method == "📤 Upload New Data":
        st.markdown("### 📤 Upload New Data for Prediction")
        new_file = st.file_uploader("Upload new CSV file for predictions", type=['csv'], key="pred_file")
        if new_file is not None:
            try:
                new_df = pd.read_csv(new_file)
                st.markdown("### 📋 Data Preview")
                st.dataframe(new_df.head(), use_container_width=True)

                if st.button("🔮 Make Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        predictions = predict_model(model, data=new_df)
                        st.success(f"✅ Predictions complete for {len(predictions)} samples!")
                        st.dataframe(predictions, use_container_width=True)
                        csv = predictions.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions</a>'
                        st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    elif method == "✍️ Manual Input":
        st.markdown("### ✍️ Enter Values Manually")
        feature_cols = [col for col in st.session_state.data.columns if col != st.session_state.target_column]
        input_data = {}
        cols = st.columns(3)
        for i, col_name in enumerate(feature_cols):
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(st.session_state.data[col_name]):
                    min_val = float(st.session_state.data[col_name].min())
                    max_val = float(st.session_state.data[col_name].max())
                    mean_val = float(st.session_state.data[col_name].mean())
                    input_data[col_name] = st.number_input(col_name, min_value=min_val, max_value=max_val, value=mean_val)
                else:
                    unique_vals = st.session_state.data[col_name].unique()[:10]
                    input_data[col_name] = st.selectbox(col_name, unique_vals)
        if st.button("🔮 Predict", type="primary"):
            input_df = pd.DataFrame([input_data])
            predictions = predict_model(model, data=input_df)
            pred_value = predictions['prediction_label'][0]
            st.markdown(f"""
            <div class="success-box">
            <h3>Predicted {st.session_state.target_column}: {pred_value}</h3>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("### 📊 Predictions on Test Data")
        if st.session_state.test_data is not None:
            X_test = st.session_state.test_data['X_test']
            y_test = st.session_state.test_data['y_test']
            predictions = predict_model(model, data=X_test)
            comp_df = X_test.copy()
            comp_df['Actual'] = y_test.values
            comp_df['Predicted'] = predictions['prediction_label'].values
            st.dataframe(comp_df.head(20), use_container_width=True)
            csv = comp_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="test_predictions.csv">📥 Download Test Predictions</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No test data available. Please train a model first.")

# ---------- 导出页面 (保持不变) ----------
def export_page():
    st.markdown('<h2 class="sub-header">💾 Export Model and Results</h2>', unsafe_allow_html=True)
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first to export results.")
        if st.button("Go to Model Training"):
            st.session_state.app_page = "📐 Model Training"
            st.rerun()
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🐍 Save Model (Pickle)")
        if st.button("Export Model as Pickle"):
            try:
                save_model(st.session_state.model, 'best_model')
                with open('best_model.pkl', 'rb') as f:
                    model_bytes = f.read()
                b64 = base64.b64encode(model_bytes).decode()
                href = f'<a href="data:file/pkl;base64,{b64}" download="best_model.pkl">📥 Download model.pkl</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Model exported successfully.")
            except Exception as e:
                st.error(f"Export failed: {e}")
    with col2:
        st.markdown("#### 📊 Model Report")
        if st.button("Generate Model Report"):
            report_content = f"""
# Machine Learning Model Report

## Project Information
- Platform: No-Code ML Platform (PyCaret)
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Problem Type: {st.session_state.problem_type}
- Target Column: {st.session_state.target_column}

## Dataset Information
- Original Shape: {st.session_state.data.shape if st.session_state.data else 'N/A'}
- Features: {len(st.session_state.data.columns) - 1 if st.session_state.data else 'N/A'}

## Model Information
- Best Model: {st.session_state.model}
- Training Completed: {st.session_state.training_complete}

## Notes
This model was generated using PyCaret AutoML through the No-Code ML Platform.
"""
            st.code(report_content, language='markdown')
            st.download_button("📥 Download Report", data=report_content,
                            file_name="ml_model_report.md", mime="text/markdown")

    st.markdown("### 📋 Session Information")
    session_info = {
        "Data Loaded": st.session_state.data is not None,
        "Target Column": st.session_state.target_column,
        "Problem Type": st.session_state.problem_type,
        "Model Trained": st.session_state.training_complete,
        "Predictions Made": st.session_state.predictions is not None,
        "Test Data Available": st.session_state.test_data is not None
    }
    session_df = pd.DataFrame.from_dict(session_info, orient='index', columns=['Status'])
    st.dataframe(session_df, use_container_width=True)

    st.markdown("### 🔄 Reset Platform")
    st.warning("This will clear all data and models from the current session.")
    if st.button("🔄 Reset All Data", type="secondary"):
        keys = ["data", "target_column", "problem_type", "model", "experiment", "predictions", "test_data", "training_complete"]
        for key in keys:
            if key in st.session_state:
                st.session_state[key] = None
        st.rerun()

# ---------- 仪表盘 Dashboard ----------
def dashboard_page():
    set_bg_image_local("purple.jpg")
    
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background:#ffffe0 !important;
    }
    section[data-testid="stSidebar"] .css-1d391kg {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] .st-emotion-cache-1wrcr25, 
    section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1 style='color: white;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
        st.markdown("### Sequential Steps")
        app_page_options = [
            "📁 Data Upload",
            "🔍 Exploratory Analysis",
            "📐 Model Training",
            "📈 Model Evaluation",
            "🔮 Make Predictions",
            "💾 Export Results"
        ]
        selected = st.radio("Select a step:", app_page_options, index=app_page_options.index(st.session_state.app_page))
        st.session_state.app_page = selected

        st.markdown("---")
        st.markdown("### Platform Info")
        st.info("""
        This platform enables:
        - CSV data upload
        - Automated EDA
        - AutoML with PyCaret
        - Model evaluation
        - No-code predictions
        """)
        if not pycaret_available:
            st.error("⚠️ PyCaret not installed. Install with: `pip install pycaret`")
            st.code("pip install pycaret", language="bash")

        if st.button("👋🏻 Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            keys = ["data", "target_column", "problem_type", "model", "experiment", "predictions", "test_data", "training_complete"]
            for key in keys:
                if key in st.session_state:
                    st.session_state[key] = None
            go_to("front")
            st.rerun()

    if st.session_state.app_page == "📁 Data Upload":
        upload_page()
    elif st.session_state.app_page == "🔍 Exploratory Analysis":
        eda_page()
    elif st.session_state.app_page == "📐 Model Training":
        training_page()
    elif st.session_state.app_page == "📈 Model Evaluation":
        evaluation_page()
    elif st.session_state.app_page == "🔮 Make Predictions":
        prediction_page()
    elif st.session_state.app_page == "💾 Export Results":
        export_page()

# ---------- 主路由 ----------
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