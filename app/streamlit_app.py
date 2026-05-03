import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import sys
import os

# Add root directory to sys.path to allow importing src locally for visualization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.kmeans_baseline import run_kmeans
    from src.enhanced_kmeans import enhanced_kmeans
    from src.data_loader import load_and_preprocess
except ImportError:
    st.error("Could not import ML modules. Make sure the app is run from the project root.")

# --- Config ---
st.set_page_config(page_title="Clustering & Outlier Detection", layout="wide", page_icon="🧩")

BASE_URL = "http://127.0.0.1:8000"

# --- Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-sub {
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.title("🧩 Clustering & Outlier Detection")
st.sidebar.markdown("Compare Baseline K-Means vs Enhanced K-Means")
st.sidebar.divider()

st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv", "txt"])

st.sidebar.header("2. Select Algorithm")
algorithm = st.sidebar.selectbox(
    "Choose clustering approach",
    ["Baseline K-Means", "Enhanced K-Means (Outlier Detection)"]
)

k_value = st.sidebar.number_input("Number of Clusters (k)", min_value=2, max_value=20, value=3, step=1)

st.sidebar.header("3. Run Experiment")
st.sidebar.markdown("Click the button below to start clustering")
run_button = st.sidebar.button("▶ Run Experiment", type="primary", use_container_width=True)

# --- State ---
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = None
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Main Area ---
st.title("Clustering & Outlier Detection")

tab1, tab2 = st.tabs(["🏠 Home", "📈 Iteration Analysis"])

with tab1:
    if uploaded_file is not None:
        # 1. Dataset Overview
        st.subheader("📄 Dataset Overview")
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        # Display dataset stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        col3.metric("Numerical Columns", num_cols)
        col4.metric("Categorical Columns", df.shape[1] - num_cols)
        
        st.markdown("**Preview (First 5 Rows)**")
        st.dataframe(df.head(), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error reading dataset: {e}")

    # Handle Run
    if run_button:
        use_enhanced = "Enhanced" in algorithm
        
        with st.spinner("Uploading and running experiment..."):
            try:
                # Step 1: Upload file
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                upload_res = requests.post(f"{BASE_URL}/datasets/upload-dataset/", files=files)
                upload_res.raise_for_status()
                file_path = upload_res.json().get("file_path")
                st.session_state.file_path = file_path
                
                # Step 2: Run experiment
                run_payload = {
                    "file_path": file_path,
                    "use_enhanced": use_enhanced,
                    "k": k_value
                }
                run_res = requests.post(f"{BASE_URL}/experiments/run", json=run_payload)
                run_res.raise_for_status()
                run_id = run_res.json().get("run_id")
                st.session_state.run_id = run_id
                
                # Step 3: Fetch results
                time.sleep(0.5) # tiny delay to ensure DB commit is visible
                res_get = requests.get(f"{BASE_URL}/experiments/{run_id}")
                res_get.raise_for_status()
                st.session_state.results = res_get.json()
                
                st.sidebar.success(f"Experiment completed successfully!\n\nRun ID: {run_id}")
                
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"API Error: {e}")
                if e.response is not None:
                    st.sidebar.error(e.response.text)
            except Exception as e:
                st.sidebar.error(f"Unexpected Error: {e}")

if st.session_state.results:
    st.divider()
    res = st.session_state.results
    
    col_results, col_plot = st.columns([1, 2])
    
    with col_results:
        st.subheader("📊 Results Summary")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 15px;">
                <div class="metric-label">Silhouette Score</div>
                <div class="metric-value">{res.get('silhouette_score', 0):.3f}</div>
                <div class="metric-sub" style="color: #059669;">Higher is better</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s2:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 15px;">
                <div class="metric-label">Davies-Bouldin Index</div>
                <div class="metric-value">{res.get('davies_bouldin_score', 0):.3f}</div>
                <div class="metric-sub" style="color: #059669;">Lower is better</div>
            </div>
            """, unsafe_allow_html=True)
        
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            if res.get('algorithm') == 'enhanced':
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-label" style="color: #7c3aed;">Number of Outliers</div>
                    <div class="metric-value" style="color: #7c3aed;">{res.get('num_outliers', 0)}</div>
                    <div class="metric-sub">Detected outliers</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-label">Number of Outliers</div>
                    <div class="metric-value" style="color: #9ca3af;">N/A</div>
                    <div class="metric-sub">Baseline Model</div>
                </div>
                """, unsafe_allow_html=True)
                
        with col_s4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Algorithm Used</div>
                <div class="metric-value" style="font-size: 16px; margin-top: 20px;">{res.get('algorithm', '').title()} K-Means</div>
            </div>
            """, unsafe_allow_html=True)

        col_s5, col_s6 = st.columns(2)
        with col_s5:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 15px;">
                <div class="metric-label">Variance</div>
                <div class="metric-value">{res.get('variance', 0):.4f}</div>
                <div class="metric-sub">Intra-cluster Variance</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s6:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 15px;">
                <div class="metric-label">Iterations</div>
                <div class="metric-value">{res.get('iterations', 1)}</div>
                <div class="metric-sub">Algorithm Cycles</div>
            </div>
            """, unsafe_allow_html=True)
            
        if res.get('algorithm') == 'enhanced':
            st.info("ℹ️ Enhanced K-Means detected outliers and returned improved clustering quality on this dataset.")

    with col_plot:
        st.subheader("📉 Cluster Visualization (PCA - 2D)")
        
        if st.session_state.file_path and os.path.exists(st.session_state.file_path):
            with st.spinner("Generating visualization..."):
                try:
                    # Run ML pipeline locally for visualization since API doesn't return full labels array
                    df_clean = load_and_preprocess(st.session_state.file_path)
                    X = df_clean.values
                    
                    if res.get('algorithm') == 'enhanced':
                        ml_res = enhanced_kmeans(X, k=k_value)
                        labels = ml_res["labels"]
                    else:
                        ml_res = run_kmeans(X, k=k_value)
                        labels = ml_res["labels"]
                        
                    # PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X)
                    
                    plot_df = pd.DataFrame({
                        "PCA Component 1": X_pca[:, 0],
                        "PCA Component 2": X_pca[:, 1],
                        "Cluster": labels
                    })
                    
                    # Convert labels for plotting
                    plot_df["Cluster Name"] = plot_df["Cluster"].apply(
                        lambda x: "Outliers" if x == -1 else f"Cluster {x}"
                    )
                    
                    # Custom color map
                    color_map = {"Outliers": "#ef4444"} # Red for outliers
                    # Colors similar to the inspiration image
                    cluster_colors = ["#3b82f6", "#f97316", "#22c55e", "#a855f7", "#ec4899", "#eab308"]
                    for i in range(10): 
                        color_map[f"Cluster {i}"] = cluster_colors[i % len(cluster_colors)]
                        
                    fig = px.scatter(
                        plot_df, 
                        x="PCA Component 1", 
                        y="PCA Component 2", 
                        color="Cluster Name",
                        color_discrete_map=color_map,
                        opacity=0.8
                    )
                    
                    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color='white')))
                    fig.update_layout(
                        legend_title_text='',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='#f3f4f6', zerolinecolor='#e5e7eb'),
                        yaxis=dict(showgrid=True, gridcolor='#f3f4f6', zerolinecolor='#e5e7eb'),
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate plot: {str(e)}")
        else:
            st.warning("Visualization not available. File path missing.")

    if not uploaded_file:
        st.info("👈 Please upload a dataset from the sidebar to get started.")

with tab2:
    if st.session_state.results and st.session_state.results.get("history"):
        st.subheader("📈 Algorithm Iteration Analysis")
        
        hist_df = pd.DataFrame(st.session_state.results["history"])
        
        with st.expander("View Raw Iteration Data"):
            st.dataframe(hist_df, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_var = px.line(hist_df, x="iteration", y="variance", markers=True, 
                              title="Intra-cluster Variance vs Iteration",
                              color_discrete_sequence=["#ef4444"])
            st.plotly_chart(fig_var, use_container_width=True)
            
            fig_sil = px.line(hist_df, x="iteration", y="silhouette", markers=True, 
                              title="Silhouette Score vs Iteration",
                              color_discrete_sequence=["#3b82f6"])
            st.plotly_chart(fig_sil, use_container_width=True)
            
        with col_p2:
            fig_db = px.line(hist_df, x="iteration", y="davies_bouldin", markers=True, 
                             title="Davies-Bouldin Index vs Iteration",
                             color_discrete_sequence=["#22c55e"])
            st.plotly_chart(fig_db, use_container_width=True)
    else:
        st.info("Run an experiment first to see iteration analysis.")

