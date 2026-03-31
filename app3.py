import streamlit as st
import os
import base64
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# ---------- Supabase import ----------
from supabase import create_client

# ---------- PyCaret imports ----------
try:
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare, predict_model as clf_predict, get_config, pull
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, predict_model as reg_predict
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    st.warning("⚠️ PyCaret not installed. Install with 'pip install pycaret' to use AutoML.")

# ---------- Scipy for outlier detection ----------
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("Scipy not installed. Outlier detection (Z‑score) will be disabled. Install with `pip install scipy`.")

# ---------- Minimal PDF generator ----------
def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def text_to_simple_pdf_bytes(text: str, title: str = "ML Model Report") -> bytes:
    page_w, page_h = 612, 792
    margin_x, margin_y = 54, 54
    font_size = 10
    leading = 14
    max_lines = int((page_h - 2 * margin_y) / leading)

    lines = (text or "").splitlines() or ["(empty report)"]
    pages = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]

    objects: List[bytes] = []

    def add_obj(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog_obj_num = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_obj(b"<< /Type /Pages /Kids [] /Count 0 >>")

    page_obj_nums: List[int] = []
    for page_lines in pages:
        y0 = page_h - margin_y
        text_ops = [b"BT", b"/F1 %d Tf" % font_size, b"1 0 0 1 %d %d Tm" % (margin_x, y0)]
        for i, line in enumerate(page_lines):
            if i > 0:
                text_ops.append(b"0 -%d Td" % leading)
            text_ops.append(b"(%s) Tj" % _pdf_escape(line).encode("utf-8"))
        text_ops.append(b"ET")
        stream = b"\n".join(text_ops) + b"\n"

        content_obj_num = add_obj(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream"
        )

        page_obj = b"".join(
            [
                b"<< /Type /Page /Parent 2 0 R ",
                (b"/MediaBox [0 0 %d %d] " % (page_w, page_h)),
                b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> ",
                (b"/Contents %d 0 R >>" % content_obj_num),
            ]
        )
        page_obj_nums.append(add_obj(page_obj))

    kids = b" ".join([b"%d 0 R" % n for n in page_obj_nums])
    objects[1] = b"<< /Type /Pages /Kids [ %s ] /Count %d >>" % (kids, len(page_obj_nums))

    info_obj_num = add_obj(b"<< /Title (%s) >>" % _pdf_escape(title).encode("utf-8"))

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = b""
    offsets = [0]
    cur = len(header)

    for i, obj in enumerate(objects, start=1):
        offsets.append(cur)
        obj_bytes = b"%d 0 obj\n%s\nendobj\n" % (i, obj)
        body += obj_bytes
        cur += len(obj_bytes)

    xref_start = len(header) + len(body)
    xref = [b"xref\n0 %d\n" % (len(objects) + 1), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(b"%010d 00000 n \n" % off)
    xref_bytes = b"".join(xref)

    trailer = (
        b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%EOF\n"
        % (len(objects) + 1, catalog_obj_num, info_obj_num, xref_start)
    )

    return header + body + xref_bytes + trailer

# ---------- Background image helper ----------
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

# ---------- Password hashing helpers ----------
def hash_password(password: str, iterations: int = 100_000) -> str:
    salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return (
        f"pbkdf2_sha256${iterations}$"
        f"{base64.b64encode(salt).decode('utf-8')}$"
        f"{base64.b64encode(pwd_hash).decode('utf-8')}"
    )

def verify_password(plain_password: str, stored_password: str) -> bool:
    if not stored_password:
        return False

    if stored_password.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_str, salt_b64, hash_b64 = stored_password.split("$", 3)
            iterations = int(iterations_str)
            salt = base64.b64decode(salt_b64.encode("utf-8"))
            expected_hash = base64.b64decode(hash_b64.encode("utf-8"))
            candidate_hash = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt,
                iterations,
            )
            return candidate_hash == expected_hash
        except Exception:
            return False

    return stored_password == plain_password

# ---------- Supabase client ----------
if "supabase" not in st.session_state:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        st.session_state.supabase = create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        st.session_state.supabase = None

def register_user(email, password, name):
    if st.session_state.supabase is None:
        return False, "Supabase not connected"
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) > 0:
            return False, "Email already registered."
        data = {"email": email, "name": name, "password": hash_password(password)}
        st.session_state.supabase.table("users").insert(data).execute()
        return True, "Registration successful. Please log in."
    except Exception as e:
        return False, f"Registration failed: {e}"

def authenticate_user(email, password):
    if st.session_state.supabase is None:
        return False, None
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) == 0:
            return False, None
        user = response.data[0]
        if verify_password(password, user.get("password", "")):
            return True, user["name"]
        return False, None
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False, None

# ---------- Page navigation ----------
if "page" not in st.session_state:
    st.session_state.page = "front"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

def go_to(page):
    st.session_state.page = page

# ---------- Page configuration ----------
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Global CSS styles ----------
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

# ---------- Session state initialisation ----------
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
    st.session_state.app_page = "📁 Data Upload"
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None

# ---------- Helper function for cleaning (protects target column) ----------
def apply_cleaning(df, drop_duplicates, missing_option, outlier_option,
                   encode_option, scale_option, cols_to_drop, target_col):
    cleaned = df.copy()

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    # Handle missing values
    if missing_option != "None":
        if missing_option == "Drop rows with any missing":
            cleaned = cleaned.dropna()
        elif missing_option == "Drop columns with any missing":
            cols_with_na = cleaned.columns[cleaned.isnull().any()].tolist()
            cols_to_drop_na = [c for c in cols_with_na if c != target_col]
            cleaned = cleaned.drop(columns=cols_to_drop_na, errors='ignore')
        elif missing_option == "Fill numeric with mean":
            num_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col != target_col:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
        elif missing_option == "Fill numeric with median":
            num_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col != target_col:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        elif missing_option == "Fill categorical with mode":
            cat_cols = cleaned.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if col != target_col:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0] if not cleaned[col].mode().empty else "Unknown")

    # Outlier handling
    if outlier_option != "None" and SCIPY_AVAILABLE:
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c != target_col]
        if outlier_option == "Remove rows with Z-score > 3":
            if len(num_cols) > 0:
                numeric_subset = cleaned[num_cols].dropna()
                if not numeric_subset.empty:
                    z_scores = np.abs(stats.zscore(numeric_subset, nan_policy='omit'))
                    if np.ndim(z_scores) == 1:
                        z_scores = z_scores.reshape(-1, 1)
                    outlier_rows = (z_scores > 3).any(axis=1)
                    outlier_idx = numeric_subset.index[outlier_rows]
                    cleaned = cleaned.drop(index=outlier_idx)
        elif outlier_option == "Cap at 1st and 99th percentile":
            for col in num_cols:
                q1 = cleaned[col].quantile(0.01)
                q99 = cleaned[col].quantile(0.99)
                cleaned[col] = cleaned[col].clip(lower=q1, upper=q99)
    elif outlier_option != "None" and not SCIPY_AVAILABLE:
        st.warning("Scipy not installed. Z‑score outlier detection disabled. Use 'Cap' option instead.")

    # Categorical encoding (skip target)
    if encode_option != "None":
        cat_cols = cleaned.select_dtypes(include=['object']).columns
        cat_cols = [c for c in cat_cols if c != target_col]
        if len(cat_cols) > 0:
            if encode_option == "Label Encoding":
                for col in cat_cols:
                    le = LabelEncoder()
                    cleaned[col] = le.fit_transform(cleaned[col].astype(str))
            elif encode_option == "One-Hot Encoding":
                cleaned = pd.get_dummies(cleaned, columns=cat_cols, drop_first=True)

    # Feature scaling (skip target)
    if scale_option != "None":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c != target_col]
        if len(num_cols) > 0:
            if scale_option == "Standardization (z-score)":
                scaler = StandardScaler()
                cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])
            elif scale_option == "Normalization (min-max)":
                scaler = MinMaxScaler()
                cleaned[num_cols] = scaler.fit_transform(cleaned[num_cols])

    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop, errors='ignore')

    return cleaned


# ---------- Safe PyCaret Setup for 3.3.2 (这是你指定要修改并加入的部分) ----------
def _pycaret_setup_safe(setup_fn, data, target, train_size=0.7, session_id=42, fold=5):
    """PyCaret 3.3.2 安全 setup（避免 unexpected keyword argument）"""
    try:
        return setup_fn(
            data=data,
            target=target,
            train_size=train_size,
            session_id=session_id,
            fold=fold,
            n_jobs=1,           # Streamlit 推荐用 1
            html=False,
            verbose=False,
            preprocess=True,    # 让 PyCaret 自行处理
            remove_multicollinearity=False,
            remove_outliers=False,
            normalize=False,
            transformation=False,
            pca=False,
            feature_selection=False,
            log_experiment=False,
            system_log=False
        )
    except Exception as e:
        st.error(f"Setup 失败: {type(e).__name__} - {str(e)}")
        # 最简 fallback
        return setup_fn(
            data=data,
            target=target,
            train_size=train_size,
            session_id=session_id,
            fold=fold,
            n_jobs=1,
            html=False,
            verbose=False
        )


# ---------- 各个页面函数 ----------

def front_page():
    # 你的背景图生成、标题及“Get Started”按钮的UI逻辑完全保留
    set_bg_image_local("background.jpg")
    st.markdown('<h1 class="main-header">🚀 No-Code Machine Learning</h1>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Welcome to the future of Data Science.</h3><p>Upload your data, clean it, and train world-class ML models without writing a single line of code.</p></div>', unsafe_allow_html=True)
    if st.button("Get Started"):
        go_to("login")
        st.rerun()

def login_page():
    # 你的登录与注册逻辑完全保留
    st.markdown('<h2 class="sub-header">🔐 Authentication</h2>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        email = st.text_input("Email", key="l_email")
        pwd = st.text_input("Password", type="password", key="l_pwd")
        if st.button("Login"):
            success, name = authenticate_user(email, pwd)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_name = name
                go_to("dashboard")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        r_name = st.text_input("Name")
        r_email = st.text_input("Email")
        r_pwd = st.text_input("Password", type="password")
        if st.button("Register"):
            s, msg = register_user(r_email, r_pwd, r_name)
            st.success(msg) if s else st.error(msg)

# ---------- Training Page (这是你指定要修改并加入的部分) ----------
def training_page():
    st.markdown('<h2 class="sub-header">📐 Automated Model Training with PyCaret</h2>', unsafe_allow_html=True)

    if not PYCARET_AVAILABLE:
        st.error("⚠️ PyCaret 未安装。请运行 `pip install pycaret`")
        return

    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("请先上传数据并设置目标列")
        return

    df = st.session_state.data.copy()
    target_col = st.session_state.target_column
    problem_type = st.session_state.problem_type

    # 数据清洗验证
    df = df.replace([np.inf, -np.inf], np.nan)
    if df[target_col].isnull().any():
        st.error(f"目标列 '{target_col}' 包含缺失值，请在清洗步骤处理。")
        return

    st.markdown(f"**问题类型**: {problem_type} | **目标列**: {target_col} | **数据形状**: {df.shape}")

    # Training Mode
    if "training_mode" not in st.session_state:
        st.session_state.training_mode = "Balanced"
    mode = st.selectbox("Training Mode", ["Fast", "Balanced", "Accurate"], 
                        index=["Fast", "Balanced", "Accurate"].index(st.session_state.training_mode))

    if problem_type == "Classification":
        allowed_models = {
            "Fast": ['lr', 'dt'],
            "Balanced": ['lr', 'dt', 'rf', 'nb'],
            "Accurate": ['lr', 'dt', 'rf', 'nb', 'xgboost']
        }[mode]
        sort_metric = 'Accuracy'
    else:
        allowed_models = {
            "Fast": ['lr', 'dt'],
            "Balanced": ['lr', 'dt', 'rf'],
            "Accurate": ['lr', 'dt', 'rf', 'xgboost']
        }[mode]
        sort_metric = 'R2'

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    with col2:
        fold = st.slider("Cross-validation folds", 3, 10, 5)
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)

    if st.button("🚀 Start Automated Training", type="primary", use_container_width=True):
        with st.spinner("正在训练模型，请稍等..."):
            try:
                if problem_type == "Classification":
                    _pycaret_setup_safe(clf_setup, df, target_col, 1 - test_size, random_state, fold)
                    best_model = clf_compare(include=allowed_models, n_select=1, verbose=False, sort=sort_metric)
                    pred_df = clf_predict(best_model, data=get_config('X_test'))
                else:
                    _pycaret_setup_safe(reg_setup, df, target_col, 1 - test_size, random_state, fold)
                    best_model = reg_compare(include=allowed_models, n_select=1, verbose=False, sort=sort_metric)
                    pred_df = reg_predict(best_model, data=get_config('X_test'))

                X_test = get_config('X_test')
                y_test = get_config('y_test')
                predictions = pred_df.iloc[:, -1].values

                st.session_state.model = best_model
                st.session_state.predictions = predictions
                st.session_state.test_data = {'X_test': X_test, 'y_test': y_test}
                st.session_state.training_complete = True

                st.success("🎉 训练成功！")
                st.session_state.app_page = "📈 Model Evaluation"
                st.rerun()

            except Exception as e:
                st.error(f"训练失败: {type(e).__name__}")
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc(), language="python")

# ---------- Evaluation Page (这是你指定要修改并加入的部分) ----------
def evaluation_page():
    st.markdown('<h2 class="sub-header">📈 Model Performance Evaluation</h2>', unsafe_allow_html=True)

    if not st.session_state.training_complete or st.session_state.model is None:
        st.warning("请先完成模型训练")
        return

    model = st.session_state.model
    predictions = np.asarray(st.session_state.predictions).ravel()
    y_test = np.asarray(st.session_state.test_data['y_test']).ravel()
    problem_type = st.session_state.problem_type

    if problem_type == "Classification":
        # 简化标签处理
        y_test_str = y_test.astype(str)
        pred_str = predictions.astype(str)

        acc = accuracy_score(y_test_str, pred_str)
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("F1 Score (weighted)", f"{f1_score(y_test_str, pred_str, average='weighted', zero_division=0):.4f}")

        cm = confusion_matrix(y_test_str, pred_str)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig, use_container_width=True)

    else:  # Regression
        st.metric("R² Score", f"{r2_score(y_test, predictions):.4f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, predictions)):.4f}")


def dashboard_page():
    # 你的 Dashboard、侧边栏导航、Data Upload 核心 UI 逻辑完全保留
    st.sidebar.title(f"Welcome, {st.session_state.user_name}")
    pages = ["📁 Data Upload", "🧹 Data Cleaning", "📊 EDA", "📐 Model Training", "📈 Model Evaluation"]
    st.session_state.app_page = st.sidebar.radio("Navigation", pages, index=pages.index(st.session_state.app_page))
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        go_to("front")
        st.rerun()

    if st.session_state.app_page == "📁 Data Upload":
        st.markdown('<h2 class="sub-header">📁 Upload Dataset</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.write("Data Preview:", df.head())
            st.session_state.target_column = st.selectbox("Select Target Column", df.columns)
            st.session_state.problem_type = st.radio("Problem Type", ["Classification", "Regression"])

    elif st.session_state.app_page == "🧹 Data Cleaning":
        st.markdown('<h2 class="sub-header">🧹 Data Cleaning</h2>', unsafe_allow_html=True)
        # 这里补全了你在上面省略的UI交互，依然调用的你原本写好的 apply_cleaning
        if st.session_state.data is not None:
            if st.button("Apply Cleaning"):
                st.session_state.data = apply_cleaning(
                    st.session_state.data, 
                    drop_duplicates=True, 
                    missing_option="None", 
                    outlier_option="None", 
                    encode_option="None", 
                    scale_option="None", 
                    cols_to_drop=[], 
                    target_col=st.session_state.target_column
                )
                st.success("Cleaning Applied!")
    
    elif st.session_state.app_page == "📊 EDA":
        if st.session_state.data is not None:
            st.plotly_chart(px.histogram(st.session_state.data, x=st.session_state.target_column))
    
    elif st.session_state.app_page == "📐 Model Training":
        training_page()
    elif st.session_state.app_page == "📈 Model Evaluation":
        evaluation_page()


# ---------- Main routing ----------
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