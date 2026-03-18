import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
import warnings
import re
import bcrypt
import pickle
from io import BytesIO, StringIO
import sys
from supabase import create_client

warnings.filterwarnings('ignore')

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
        st.error(f"Supabase 连接失败: {e}")
        st.session_state.supabase = None

# ---------- 背景图片处理 ----------
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
        # 后备纯色背景
        fallback_bg = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """
        st.markdown(fallback_bg, unsafe_allow_html=True)

# ---------- 用户认证 (密码哈希) ----------
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def register_user(email, password, name):
    if st.session_state.supabase is None:
        return False, "Supabase 未连接"
    if not validate_email(email):
        return False, "邮箱格式无效"
    try:
        # 检查邮箱是否已存在
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) > 0:
            return False, "该邮箱已注册"
        # 密码哈希
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        data = {"email": email, "name": name, "password": hashed}
        st.session_state.supabase.table("users").insert(data).execute()
        return True, "注册成功，请登录"
    except Exception as e:
        return False, f"注册失败: {e}"

def authenticate_user(email, password):
    if st.session_state.supabase is None:
        return False, None
    try:
        response = st.session_state.supabase.table("users").select("*").eq("email", email).execute()
        if len(response.data) == 0:
            return False, None
        user = response.data[0]
        if bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
            return True, user["name"]
        else:
            return False, None
    except Exception as e:
        st.error(f"认证失败: {e}")
        return False, None

# ---------- 会话状态初始化 ----------
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
if "model" not in st.session_state:          # 存储训练好的TPOT对象
    st.session_state.model = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "X_columns" not in st.session_state:      # 原始特征列名
    st.session_state.X_columns = None
if "cat_cols" not in st.session_state:       # 分类列名
    st.session_state.cat_cols = None
if "num_cols" not in st.session_state:       # 数值列名
    st.session_state.num_cols = None
if "encoded_columns" not in st.session_state: # 编码后的特征列名
    st.session_state.encoded_columns = None
if "num_impute_values" not in st.session_state: # 数值列填充值
    st.session_state.num_impute_values = None
if "cat_impute_values" not in st.session_state: # 分类列填充值
    st.session_state.cat_impute_values = None

def go_to(page):
    st.session_state.page = page

# ---------- 全局CSS样式 ----------
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
    /* 登录页表单样式 */
    .form-card {
        background: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 10px;
        color: white;
    }
    .stTextInput label {
        color: white !important;
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
</style>
""", unsafe_allow_html=True)

# ---------- 首页 ----------
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
                <p style="color: white;">未找到动画文件 animation.mp4</p>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="right-panel">
            <h1>Welcome to<br>No-Code ML Platform</h1>
            <p>Accessible for Machine Learning without code.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Get Started", key="get_started", use_container_width=True):
            go_to("login")
            st.rerun()

# ---------- 登录/注册页面 ----------
def login_page():
    set_bg_image_local("login.jpg")
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
                        st.error("邮箱或密码错误")
        
        with tab2:
            with st.form("register_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register")
                if submitted:
                    if password != confirm:
                        st.error("两次输入的密码不一致")
                    elif len(password) < 6:
                        st.error("密码至少6位")
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

# ---------- Dashboard 页面 ----------
def dashboard_page():
    set_bg_image_local("purple.jpg")
    # 侧边栏样式
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background:#ffffe0 !important;
    }
    section[data-testid="stSidebar"] .st-emotion-cache-1wrcr25, 
    section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1 style='color: white;'>Welcome, {st.session_state.user_name}!</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
        st.markdown("### ML Workflow")
        step = st.sidebar.radio(
            "Go to",
            ["📁 Data Upload", "🔍 Data Overview & Target", "🤖 TPOT Training", "📊 Results & Predictions"],
            key="nav_step"
        )

        if st.button("👋🏻 Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            keys = ["data", "target_column", "problem_type", "model", "training_complete",
                    "X_columns", "cat_cols", "num_cols", "encoded_columns",
                    "num_impute_values", "cat_impute_values"]
            for key in keys:
                if key in st.session_state:
                    st.session_state[key] = None
            go_to("front")
            st.rerun()

    # ---------- Step 1: Data Upload ----------
    def data_upload_step():
        st.header("📁 Upload your dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.data = df
                # 重置所有下游状态
                st.session_state.training_complete = False
                st.session_state.model = None
                st.session_state.target_column = None
                st.session_state.X_columns = None
                st.session_state.cat_cols = None
                st.session_state.num_cols = None
                st.session_state.encoded_columns = None
                st.session_state.num_impute_values = None
                st.session_state.cat_impute_values = None
                st.success("文件加载成功！")
                st.dataframe(df.head())
                st.write(f"数据维度: {df.shape[0]} 行, {df.shape[1]} 列")
            except Exception as e:
                st.error(f"文件加载失败: {e}")
        else:
            if st.session_state.data is not None:
                st.info("使用已上传的数据。")
                st.dataframe(st.session_state.data.head())
            else:
                st.info("请上传一个CSV或Excel文件开始。")

    # ---------- Step 2: Data Overview & Target Selection ----------
    def data_overview_step():
        if st.session_state.data is None:
            st.warning("请先上传数据。")
            return
        df = st.session_state.data
        st.subheader("🔍 Data Overview")
        st.write("前几行数据：")
        st.dataframe(df.head())
        st.write("数据类型：")
        st.write(df.dtypes)

        target = st.selectbox("选择目标列", df.columns.tolist(), index=None)
        if target:
            st.session_state.target_column = target
            # 自动推断问题类型
            y = df[target]
            if y.dtype in ['object', 'category'] or y.nunique() < 10:
                default_type = "classification"
            else:
                default_type = "regression"
            problem_type = st.radio(
                "问题类型",
                ["classification", "regression"],
                index=0 if default_type == "classification" else 1
            )
            st.session_state.problem_type = problem_type
            st.session_state.X_columns = [col for col in df.columns if col != target]
            st.write(f"特征列 ({len(st.session_state.X_columns)}): {st.session_state.X_columns}")
            # 目标列变更后，训练状态失效
            st.session_state.training_complete = False
            st.session_state.model = None
        else:
            st.session_state.target_column = None
            st.session_state.X_columns = None

    # ---------- Step 3: TPOT Training (带自动编码和填充) ----------
    def tpot_training_step():
        if st.session_state.data is None or st.session_state.target_column is None:
            st.warning("请先上传数据并选择目标列。")
            return
        st.header("🤖 TPOT AutoML Training")
        st.write(f"目标列: **{st.session_state.target_column}** ({st.session_state.problem_type})")
        st.write(f"特征列: {len(st.session_state.X_columns)} 个")

        # 数据量警告
        if st.session_state.data.shape[0] > 50000:
            st.warning("数据量较大（超过5万行），训练可能非常耗时。建议使用采样或耐心等待。")

        # TPOT 参数
        generations = st.number_input("进化代数", min_value=1, max_value=100, value=5, step=1)
        population_size = st.number_input("种群大小", min_value=1, max_value=100, value=10, step=1)
        cv = st.number_input("交叉验证折数", min_value=2, max_value=10, value=5, step=1)
        
        if st.session_state.problem_type == "classification":
            scoring_options = ["accuracy", "f1", "roc_auc", "average_precision"]
        else:
            scoring_options = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
        scoring = st.selectbox("评分指标", scoring_options)
        
        verbosity = st.slider("日志详细程度", 0, 3, 2)
        random_state = st.number_input("随机种子", value=42)

        # 缺失值处理选项
        handle_missing = st.checkbox("自动填充缺失值（数值列用均值，分类列用众数）", value=True)

        if st.button("🚀 开始训练", type="primary"):
            with st.spinner("TPOT 正在优化管道... 可能需要几分钟。"):
                df = st.session_state.data
                target = st.session_state.target_column

                # 删除目标列缺失的行
                df_clean = df.dropna(subset=[target])
                if len(df_clean) < len(df):
                    st.info(f"删除了 {len(df)-len(df_clean)} 行目标缺失的数据。")

                y = df_clean[target]
                X = df_clean.drop(columns=[target])

                # 识别分类和数值列
                cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                st.write(f"分类特征: {cat_cols}")
                st.write(f"数值特征: {num_cols}")

                # 保存列信息供预测使用
                st.session_state.cat_cols = cat_cols
                st.session_state.num_cols = num_cols

                # 处理缺失值（如果选择）
                if handle_missing:
                    # 数值列用均值填充
                    num_impute_values = X[num_cols].mean() if num_cols else pd.Series()
                    X[num_cols] = X[num_cols].fillna(num_impute_values)
                    st.session_state.num_impute_values = num_impute_values.to_dict() if num_cols else {}

                    # 分类列用众数填充
                    cat_impute_values = {}
                    for col in cat_cols:
                        mode_val = X[col].mode()
                        if len(mode_val) > 0:
                            fill_val = mode_val[0]
                        else:
                            fill_val = "missing"
                        X[col] = X[col].fillna(fill_val)
                        cat_impute_values[col] = fill_val
                    st.session_state.cat_impute_values = cat_impute_values

                # 对分类列进行独热编码
                if cat_cols:
                    X_encoded = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
                else:
                    X_encoded = X.copy()

                # 记录编码后的列名（用于预测时对齐）
                encoded_columns = X_encoded.columns.tolist()
                st.session_state.encoded_columns = encoded_columns

                # 实例化 TPOT
                if st.session_state.problem_type == "classification":
                    tpot = TPOTClassifier(
                        generations=generations,
                        population_size=population_size,
                        cv=cv,
                        scoring=scoring,
                        verbosity=verbosity,
                        random_state=random_state,
                        n_jobs=-1
                    )
                else:
                    tpot = TPOTRegressor(
                        generations=generations,
                        population_size=population_size,
                        cv=cv,
                        scoring=scoring,
                        verbosity=verbosity,
                        random_state=random_state,
                        n_jobs=-1
                    )

                # 捕获训练日志
                log_output = StringIO()
                old_stdout = sys.stdout
                sys.stdout = log_output

                try:
                    tpot.fit(X_encoded, y)
                except Exception as e:
                    st.error(f"TPOT 训练失败: {e}")
                    sys.stdout = old_stdout
                    st.exception(e)
                    return
                finally:
                    sys.stdout = old_stdout

                # 显示日志
                with st.expander("训练日志", expanded=True):
                    st.text(log_output.getvalue())

                st.success("训练完成！")
                st.session_state.model = tpot
                st.session_state.training_complete = True

                # 显示最佳管道和得分
                st.subheader("最佳管道")
                st.code(str(tpot.fitted_pipeline_))

                if hasattr(tpot, '_optimized_pipeline_score'):
                    st.metric("最佳交叉验证得分", f"{tpot._optimized_pipeline_score:.4f}")

    # ---------- Step 4: Results & Predictions ----------
    def results_step():
        if not st.session_state.training_complete or st.session_state.model is None:
            st.warning("请先训练模型。")
            return
        st.header("📊 结果与预测")

        tpot = st.session_state.model
        pipeline = tpot.fitted_pipeline_

        st.subheader("最佳管道")
        st.code(str(pipeline))

        # 下载模型
        st.subheader("下载模型")
        if pipeline:
            buf = BytesIO()
            pickle.dump(pipeline, buf)
            buf.seek(0)
            st.download_button(
                label="📥 下载模型 (pickle)",
                data=buf,
                file_name="tpot_model.pkl",
                mime="application/octet-stream"
            )

        # 对新数据进行预测
        st.subheader("对新数据进行预测")
        pred_file = st.file_uploader("上传新数据 (CSV) 进行预测", type=["csv"], key="pred_upload")
        if pred_file is not None:
            try:
                new_df = pd.read_csv(pred_file)
                st.dataframe(new_df.head())

                # 检查必要的列是否存在
                missing_cols = set(st.session_state.X_columns) - set(new_df.columns)
                if missing_cols:
                    st.error(f"新数据缺少以下列: {missing_cols}")
                    return

                X_new = new_df[st.session_state.X_columns].copy()

                # 应用与训练相同的预处理
                # 1. 缺失值填充
                if st.session_state.num_impute_values:
                    for col, val in st.session_state.num_impute_values.items():
                        if col in X_new.columns:
                            X_new[col] = X_new[col].fillna(val)
                if st.session_state.cat_impute_values:
                    for col, val in st.session_state.cat_impute_values.items():
                        if col in X_new.columns:
                            X_new[col] = X_new[col].fillna(val)

                # 2. 对分类列进行独热编码，并与训练集列对齐
                cat_cols = st.session_state.cat_cols or []
                if cat_cols:
                    X_encoded = pd.get_dummies(X_new, columns=cat_cols, dummy_na=False)
                else:
                    X_encoded = X_new.copy()

                # 对齐训练集列名：缺失的列补0，多余的列丢弃
                encoded_columns = st.session_state.encoded_columns
                for col in encoded_columns:
                    if col not in X_encoded.columns:
                        X_encoded[col] = 0
                X_encoded = X_encoded[encoded_columns]  # 保持列顺序与训练一致

                # 预测
                predictions = pipeline.predict(X_encoded)
                st.write("预测结果：")
                st.write(predictions)

                # 分类概率
                if st.session_state.problem_type == "classification" and st.checkbox("显示概率"):
                    if hasattr(pipeline, "predict_proba"):
                        probs = pipeline.predict_proba(X_encoded)
                        st.write("预测概率：")
                        st.write(probs)
                    else:
                        st.info("当前管道不支持概率预测。")

                # 下载预测结果
                pred_df = pd.DataFrame({"Prediction": predictions})
                csv = pred_df.to_csv(index=False)
                st.download_button("📥 下载预测结果", csv, "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"预测失败: {e}")
                st.exception(e)

    # ---------- 路由到选中的步骤 ----------
    if step == "📁 Data Upload":
        data_upload_step()
    elif step == "🔍 Data Overview & Target":
        data_overview_step()
    elif step == "🤖 TPOT Training":
        tpot_training_step()
    elif step == "📊 Results & Predictions":
        results_step()

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