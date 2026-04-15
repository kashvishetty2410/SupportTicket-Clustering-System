import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import os
from collections import Counter
from scipy.sparse import load_npz, save_npz
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TicketMind — HAC Clustering",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME  (light / dark toggle stored in session)
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

DARK = st.session_state.dark_mode

# ── colour tokens ──
if DARK:
    BG        = "#0b0d0f"
    SURFACE   = "#12151a"
    SURFACE2  = "#191d24"
    TEXT      = "#d4d8e0"
    MUTED     = "#606672"
    ACCENT    = "#4fffb0"
    BORDER    = "rgba(255,255,255,0.07)"
    MPL_BG    = "#12151a"
    MPL_FG    = "#d4d8e0"
    MPL_GRID  = "#232830"
else:
    BG        = "#f5f6f8"
    SURFACE   = "#ffffff"
    SURFACE2  = "#eef0f4"
    TEXT      = "#1a1d23"
    MUTED     = "#7a8090"
    ACCENT    = "#0d9e6a"
    BORDER    = "rgba(0,0,0,0.08)"
    MPL_BG    = "#ffffff"
    MPL_FG    = "#1a1d23"
    MPL_GRID  = "#e8eaed"

CLUSTER_COLORS = ["#7c6dfa", "#f4894b", "#fa5b6a", "#38c4f4", "#e0a644",
                  "#4fffb0", "#e05ca8", "#5bc4fa"]

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── root ── */
html, body, [class*="css"] {{
    font-family: 'Syne', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

/* ── main container ── */
[data-testid="stAppViewContainer"] > .main {{ background-color: {BG} !important; }}
[data-testid="block-container"] {{ padding: 1.6rem 2rem !important; }}

/* ── metric cards ── */
[data-testid="metric-container"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 20px !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: {ACCENT} !important;
    font-size: 2rem !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: {MUTED} !important;
    text-transform: uppercase;
    letter-spacing: .07em;
}}
[data-testid="stMetricDelta"] {{ font-size: 11px !important; }}

/* ── tabs ── */
[data-testid="stTabs"] button {{
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: .05em;
    color: {MUTED} !important;
    border-radius: 8px 8px 0 0 !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
    background: {SURFACE2} !important;
}}

/* ── dataframe ── */
[data-testid="stDataFrame"] {{
    background: {SURFACE} !important;
    border-radius: 12px;
    border: 1px solid {BORDER};
}}

/* ── buttons ── */
[data-testid="stButton"] > button {{
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    background: {SURFACE2} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    transition: all .2s;
}}
[data-testid="stButton"] > button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
}}

/* ── sliders / selects ── */
[data-testid="stSlider"] label, [data-testid="stSelectbox"] label,
[data-testid="stRadio"] label, .stSelectbox label {{
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: {MUTED} !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}}

/* ── text inputs ── */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {{
    background: {SURFACE2} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}}

/* ── expander ── */
[data-testid="stExpander"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}
[data-testid="stExpander"] summary {{
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    color: {TEXT} !important;
}}

/* ── divider ── */
hr {{ border-color: {BORDER} !important; }}

/* ── hide streamlit branding ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── cluster chip ── */
.chip {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    margin: 1px;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATHS  (relative to app.py location)
# ─────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data", "processed")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
RAW_CSV   = os.path.join(BASE, "data", "banks_subset.csv")

# ─────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_cleaned():
    return pd.read_csv(os.path.join(DATA_DIR, "cleaned_data.csv"))

@st.cache_data(show_spinner=False)
def load_clustered():
    return pd.read_csv(os.path.join(DATA_DIR, "clustered_data.csv"))

@st.cache_data(show_spinner=False)
def load_summary():
    return pd.read_csv(os.path.join(DATA_DIR, "cluster_summary.csv"))

@st.cache_data(show_spinner=False)
def load_tfidf():
    return load_npz(os.path.join(DATA_DIR, "tfidf_matrix.npz"))

@st.cache_data(show_spinner=False)
def load_report():
    p = os.path.join(PLOTS_DIR, "cluster_analysis_report.txt")
    with open(p, encoding="utf-8") as f:
        return f.read()

# ─────────────────────────────────────────────
# MATPLOTLIB THEME HELPER
# ─────────────────────────────────────────────
def mpl_theme():
    plt.rcParams.update({
        "figure.facecolor":  MPL_BG,
        "axes.facecolor":    MPL_BG,
        "axes.edgecolor":    MPL_GRID,
        "axes.labelcolor":   MPL_FG,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        MPL_FG,
        "grid.color":        MPL_GRID,
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
    })

# ─────────────────────────────────────────────
# RE-RUN CLUSTERING  (called from Tab 5)
# ─────────────────────────────────────────────
def rerun_clustering(n_clusters, linkage_method):
    X = load_tfidf()
    X_dense = X.toarray()
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_dense)
    df = load_cleaned().copy()
    df["cluster"] = labels
    return df, X_dense, labels

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:24px'>
      <div style='width:40px;height:40px;border-radius:12px;
           background:linear-gradient(135deg,#1ac97a,#4fffb0);
           display:flex;align-items:center;justify-content:center;font-size:20px'>🧠</div>
      <div>
        <div style='font-size:18px;font-weight:800;letter-spacing:-.02em'>
          Ticket<span style='color:{ACCENT}'>Mind</span></div>
        <div style='font-family:"Space Mono",monospace;font-size:9px;color:{MUTED};letter-spacing:.06em'>
          HAC · TF-IDF · UNSUPERVISED</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dark / Light toggle
    st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:9px;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px'>APPEARANCE</div>", unsafe_allow_html=True)
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        if st.button("🌙 Dark" if not DARK else "🌙 Dark ✓",
                     use_container_width=True):
            st.session_state.dark_mode = True
            st.rerun()
    with col_t2:
        if st.button("☀️ Light" if DARK else "☀️ Light ✓",
                     use_container_width=True):
            st.session_state.dark_mode = False
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Quick stats from summary
    try:
        summary = load_summary()
        clustered = load_clustered()
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:9px;color:{MUTED};letter-spacing:.08em;margin-bottom:10px'>DATASET STATS</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;margin-bottom:4px'>📄 <b>{len(clustered):,}</b> total tickets</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;margin-bottom:4px'>🗂️ <b>{clustered['cluster'].nunique()}</b> clusters</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;margin-bottom:4px'>🏷️ <b>{clustered['product'].nunique()}</b> products</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;margin-bottom:16px'>⚠️ <b>{clustered['issue'].nunique()}</b> unique issues</div>", unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace;font-size:9px;color:{MUTED};line-height:1.9;letter-spacing:.03em'>
      MODEL · Agglomerative HC<br>
      FEATURES · TF-IDF (3000)<br>
      REDUCTION · TruncatedSVD<br>
      LINKAGE · Ward<br>
      CLUSTERS · 4
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<h1 style='font-size:28px;font-weight:800;letter-spacing:-.02em;margin-bottom:2px'>
  HAC Customer Support Ticket Clustering
</h1>
<p style='font-family:Space Mono,monospace;font-size:11px;color:{MUTED};margin-bottom:24px'>
  Agglomerative Hierarchical Clustering · TF-IDF Features · Banks Consumer Complaints Dataset
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA (with error guard)
# ─────────────────────────────────────────────
try:
    df_clean    = load_cleaned()
    df_cluster  = load_clustered()
    df_summary  = load_summary()
    X_sparse    = load_tfidf()
except FileNotFoundError as e:
    st.error(f"❌ Could not find processed data files.\n\nMake sure you've run the preprocessing and clustering scripts first.\n\n`{e}`")
    st.stop()

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Tickets",   f"{len(df_cluster):,}")
k2.metric("Clusters Found",  df_cluster["cluster"].nunique())
k3.metric("Products",        df_cluster["product"].nunique())
k4.metric("Unique Issues",   df_cluster["issue"].nunique())
k5.metric("Avg Ticket Len",  f"{int(df_cluster['text_length'].mean() if 'text_length' in df_cluster.columns else df_clean['text_length'].mean())} chars")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview",
    "🔍  Explore Tickets",
    "🌳  Dendrogram",
    "📈  Visualisations",
    "⚙️  Re-run Model",
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;margin-bottom:16px'>CLUSTER SUMMARY</p>", unsafe_allow_html=True)

    # Summary table
    st.dataframe(
        df_summary.style.set_properties(**{
            "background-color": SURFACE,
            "color": TEXT,
            "font-family": "Space Mono, monospace",
            "font-size": "11px",
        }),
        use_container_width=True,
        height=200,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    # ── Cluster size bar chart ──
    with col_a:
        st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.07em;text-transform:uppercase'>CLUSTER SIZES</p>", unsafe_allow_html=True)
        mpl_theme()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sizes = df_cluster["cluster"].value_counts().sort_index()
        bars = ax.barh(
            [f"Cluster {i}" for i in sizes.index],
            sizes.values,
            color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in sizes.index],
            height=0.55,
        )
        for bar, val in zip(bars, sizes.values):
            ax.text(val + sizes.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9, color=MPL_FG)
        ax.set_xlabel("Number of tickets", fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Product distribution pie ──
    with col_b:
        st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.07em;text-transform:uppercase'>TOP PRODUCTS</p>", unsafe_allow_html=True)
        mpl_theme()
        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        prod_counts = df_cluster["product"].value_counts().head(5)
        wedge_colors = CLUSTER_COLORS[:len(prod_counts)]
        wedges, texts, autotexts = ax2.pie(
            prod_counts.values,
            labels=None,
            autopct="%1.0f%%",
            colors=wedge_colors,
            startangle=90,
            wedgeprops=dict(linewidth=1.5, edgecolor=MPL_BG),
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(9); at.set_color(MPL_FG)
        ax2.legend(
            wedges, [p[:28] for p in prod_counts.index],
            loc="lower center", bbox_to_anchor=(0.5, -0.25),
            ncol=2, fontsize=8, frameon=False,
            labelcolor=MPL_FG,
        )
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Top issues per cluster ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.07em;text-transform:uppercase'>TOP ISSUE PER CLUSTER</p>", unsafe_allow_html=True)

    issue_cols = st.columns(df_cluster["cluster"].nunique())
    for idx, cid in enumerate(sorted(df_cluster["cluster"].unique())):
        sub = df_cluster[df_cluster["cluster"] == cid]
        top_issue = sub["issue"].value_counts().index[0]
        top_product = sub["product"].value_counts().index[0]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        with issue_cols[idx]:
            st.markdown(f"""
            <div style='background:{SURFACE};border:1px solid {color}44;
                 border-left:3px solid {color};border-radius:12px;padding:14px 16px'>
              <div style='font-family:Space Mono,monospace;font-size:9px;color:{MUTED};
                   letter-spacing:.06em;margin-bottom:6px'>CLUSTER {cid}</div>
              <div style='font-size:13px;font-weight:700;color:{color};
                   margin-bottom:4px'>{len(sub):,} tickets</div>
              <div style='font-size:11px;color:{TEXT};margin-bottom:3px'>
                ⚠️ {top_issue[:40]}</div>
              <div style='font-size:10px;color:{MUTED}'>🏷️ {top_product[:35]}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — EXPLORE TICKETS
# ══════════════════════════════════════════════
with tab2:
    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;margin-bottom:16px'>SEARCH & FILTER CLUSTERED TICKETS</p>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search_q = st.text_input("🔍 Search ticket text", placeholder="e.g. payment failed, account blocked…")
    with fc2:
        cluster_filter = st.selectbox("Filter by cluster",
            ["All"] + [f"Cluster {i}" for i in sorted(df_cluster["cluster"].unique())])
    with fc3:
        product_filter = st.selectbox("Filter by product",
            ["All"] + sorted(df_cluster["product"].dropna().unique().tolist()))

    # apply filters
    view = df_cluster.copy()
    if search_q:
        view = view[view["consumer_complaint_narrative"].str.contains(search_q, case=False, na=False)]
    if cluster_filter != "All":
        cid = int(cluster_filter.split()[-1])
        view = view[view["cluster"] == cid]
    if product_filter != "All":
        view = view[view["product"] == product_filter]

    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED}'>{len(view):,} tickets matched</p>", unsafe_allow_html=True)

    # Display as cards (first 50)
    for _, row in view.head(50).iterrows():
        cid = int(row["cluster"])
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        snippet = str(row["consumer_complaint_narrative"])[:280].replace("\n", " ")
        product = row.get("product", "—")
        issue   = row.get("issue", "—")
        st.markdown(f"""
        <div style='background:{SURFACE};border:1px solid {BORDER};border-left:3px solid {color};
             border-radius:10px;padding:12px 16px;margin-bottom:8px'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:7px'>
            <span style='background:{color}22;color:{color};border:1px solid {color}55;
               font-family:Space Mono,monospace;font-size:9px;font-weight:700;
               padding:2px 9px;border-radius:20px'>CLUSTER {cid}</span>
            <span style='font-family:Space Mono,monospace;font-size:9px;color:{MUTED}'>{product[:40]}</span>
            <span style='font-family:Space Mono,monospace;font-size:9px;color:{MUTED};margin-left:auto'>{issue[:40]}</span>
          </div>
          <div style='font-size:12px;color:{TEXT};line-height:1.65'>{snippet}…</div>
        </div>
        """, unsafe_allow_html=True)

    if len(view) > 50:
        st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED}'>Showing first 50 of {len(view):,} results.</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — DENDROGRAM
# ══════════════════════════════════════════════
with tab3:
    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;margin-bottom:16px'>HIERARCHICAL MERGE TREE (WARD LINKAGE)</p>", unsafe_allow_html=True)

    d1, d2 = st.columns([3, 1])
    with d2:
        st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:9px;color:{MUTED};letter-spacing:.06em;text-transform:uppercase'>CONTROLS</p>", unsafe_allow_html=True)
        dend_linkage  = st.selectbox("Linkage method", ["ward", "complete", "average", "single"], key="dend_link")
        dend_truncate = st.slider("Truncation level (p)", 2, 15, 5, key="dend_p")
        dend_thresh   = st.slider("Cut threshold (%)", 10, 90, 50, key="dend_t")

        st.markdown(f"""
        <div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;
             padding:12px 14px;margin-top:16px;font-family:Space Mono,monospace;font-size:9px;
             color:{MUTED};line-height:2'>
          <b style='color:{TEXT}'>How to read:</b><br>
          🔴 Red line = cut threshold<br>
          📏 Y-axis = merge distance<br>
          🌿 Tall lines = distinct clusters<br>
          ✂️ Intersections = cluster count
        </div>
        """, unsafe_allow_html=True)

    with d1:
        with st.spinner("Computing dendrogram…"):
            try:
                n_comp = min(50, X_sparse.shape[0] - 1)
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                X_red = svd.fit_transform(X_sparse)
                Z = linkage(X_red, method=dend_linkage)

                mpl_theme()
                fig_d, ax_d = plt.subplots(figsize=(12, 5.5))

                dendrogram(
                    Z,
                    truncate_mode="level",
                    p=dend_truncate,
                    leaf_rotation=90,
                    leaf_font_size=8,
                    show_contracted=True,
                    ax=ax_d,
                    color_threshold=0,
                    above_threshold_color=MUTED,
                    link_color_func=lambda k: CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                )

                # threshold line
                dist_min, dist_max = Z[:, 2].min(), Z[:, 2].max()
                thresh_val = dist_min + (dist_max - dist_min) * (dend_thresh / 100)
                ax_d.axhline(thresh_val, color="#fa5b6a", linewidth=1.5,
                             linestyle="--", label=f"cut = {thresh_val:.2f}")
                ax_d.legend(fontsize=9, frameon=False, labelcolor=MPL_FG)

                n_cut = len(np.unique(fcluster(Z, t=thresh_val, criterion="distance")))
                ax_d.set_title(
                    f"Dendrogram  ·  {dend_linkage.capitalize()} linkage  ·  ~{n_cut} clusters at cut",
                    fontsize=11, fontweight="bold", pad=14
                )
                ax_d.set_xlabel("Sample / Cluster (size)", fontsize=9)
                ax_d.set_ylabel("Distance", fontsize=9)
                ax_d.spines[["top", "right"]].set_visible(False)
                fig_d.tight_layout()
                st.pyplot(fig_d, use_container_width=True)
                plt.close()

                st.markdown(f"""
                <div style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};
                     margin-top:8px;text-align:center'>
                  Explained variance (TruncatedSVD): <b style='color:{ACCENT}'>
                  {svd.explained_variance_ratio_.sum():.1%}</b>
                  &nbsp;|&nbsp; Merges: <b style='color:{ACCENT}'>{Z.shape[0]}</b>
                  &nbsp;|&nbsp; Clusters at cut: <b style='color:{ACCENT}'>{n_cut}</b>
                </div>
                """, unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Could not generate dendrogram: {ex}")

# ══════════════════════════════════════════════
# TAB 4 — VISUALISATIONS
# ══════════════════════════════════════════════
with tab4:
    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;margin-bottom:16px'>PRE-GENERATED PLOTS FROM MODEL PIPELINE</p>", unsafe_allow_html=True)

    plot_meta = {
        "dendrogram.png":         ("🌳 Dendrogram",              "Full dendrogram from Ward linkage"),
        "cluster_pca.png":        ("📌 PCA Cluster Plot",         "2D projection of TF-IDF clusters"),
        "cluster_plot.png":       ("⚫ Cluster Scatter",          "Scatter plot with cluster colours"),
        "cluster_distribution.png": ("📊 Cluster Distribution",  "Ticket count per cluster"),
        "top_words.png":          ("📝 Top Words",                "Most frequent terms across all tickets"),
        "top_issues.png":         ("⚠️ Top Issues",              "Most common issue types"),
        "top_products.png":       ("🏷️ Top Products",            "Most common product categories"),
        "text_length.png":        ("📏 Text Length Distribution", "Histogram of complaint lengths"),
        "wordcloud.png":          ("☁️ Word Cloud",              "Visual frequency of key terms"),
    }

    # layout: 3 columns
    plot_items = list(plot_meta.items())
    for row_start in range(0, len(plot_items), 3):
        cols = st.columns(3)
        for col, (fname, (title, desc)) in zip(cols, plot_items[row_start:row_start + 3]):
            fpath = os.path.join(PLOTS_DIR, fname)
            with col:
                st.markdown(f"""
                <div style='margin-bottom:6px'>
                  <span style='font-size:13px;font-weight:700'>{title}</span><br>
                  <span style='font-family:Space Mono,monospace;font-size:9px;color:{MUTED}'>{desc}</span>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists(fpath):
                    st.image(fpath, use_column_width=True)
                else:
                    st.markdown(f"<div style='background:{SURFACE2};border:1px dashed {BORDER};border-radius:10px;padding:28px;text-align:center;font-family:Space Mono,monospace;font-size:9px;color:{MUTED}'>File not found</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Analysis report
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px'>CLUSTER ANALYSIS REPORT</p>", unsafe_allow_html=True)
    try:
        report = load_report()
        st.code(report, language=None)
    except Exception:
        st.info("cluster_analysis_report.txt not found in plots folder.")

# ══════════════════════════════════════════════
# TAB 5 — RE-RUN MODEL
# ══════════════════════════════════════════════
with tab5:
    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;margin-bottom:16px'>EXPERIMENT WITH CLUSTERING PARAMETERS</p>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        new_n = st.slider("Number of clusters", 2, 10, 4)
    with r2:
        new_link = st.selectbox("Linkage method", ["ward", "complete", "average", "single"], key="rerun_link")
    with r3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Clustering", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running Agglomerative Clustering  (k={new_n}, linkage={new_link})…"):
            try:
                df_new, X_dense, new_labels = rerun_clustering(new_n, new_link)

                st.success(f"✅ Clustering complete — {new_n} clusters found!")

                # Distribution
                ra, rb = st.columns(2)
                with ra:
                    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.07em;text-transform:uppercase;margin-bottom:8px'>CLUSTER DISTRIBUTION</p>", unsafe_allow_html=True)
                    mpl_theme()
                    fig_r, ax_r = plt.subplots(figsize=(6, 3))
                    sizes_r = df_new["cluster"].value_counts().sort_index()
                    ax_r.bar(
                        [f"C{i}" for i in sizes_r.index],
                        sizes_r.values,
                        color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in sizes_r.index],
                        width=0.6,
                    )
                    ax_r.set_ylabel("Tickets", fontsize=9)
                    ax_r.spines[["top", "right"]].set_visible(False)
                    ax_r.grid(axis="y", alpha=0.3)
                    fig_r.tight_layout()
                    st.pyplot(fig_r, use_container_width=True)
                    plt.close()

                # PCA scatter
                with rb:
                    st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.07em;text-transform:uppercase;margin-bottom:8px'>PCA SCATTER (2D)</p>", unsafe_allow_html=True)
                    pca = PCA(n_components=2)
                    X2d = pca.fit_transform(X_dense)
                    mpl_theme()
                    fig_p, ax_p = plt.subplots(figsize=(6, 3))
                    for cid in range(new_n):
                        mask = new_labels == cid
                        ax_p.scatter(X2d[mask, 0], X2d[mask, 1],
                                     c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                                     s=4, alpha=0.5, label=f"C{cid}", rasterized=True)
                    ax_p.legend(fontsize=7, frameon=False, labelcolor=MPL_FG,
                                loc="upper right", markerscale=3)
                    ax_p.spines[["top", "right"]].set_visible(False)
                    ax_p.set_xlabel("PC1", fontsize=8); ax_p.set_ylabel("PC2", fontsize=8)
                    fig_p.tight_layout()
                    st.pyplot(fig_p, use_container_width=True)
                    plt.close()

                # Summary table
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:.07em;text-transform:uppercase'>CLUSTER BREAKDOWN</p>", unsafe_allow_html=True)
                summary_rows = []
                for cid in sorted(df_new["cluster"].unique()):
                    sub = df_new[df_new["cluster"] == cid]
                    summary_rows.append({
                        "Cluster": cid,
                        "Size": len(sub),
                        "% of Total": f"{len(sub)/len(df_new)*100:.1f}%",
                        "Top Product": sub["product"].value_counts().index[0] if "product" in sub else "—",
                        "Top Issue": sub["issue"].value_counts().index[0] if "issue" in sub else "—",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            except Exception as ex:
                st.error(f"Clustering failed: {ex}")
    else:
        st.markdown(f"""
        <div style='background:{SURFACE};border:1px dashed {BORDER};border-radius:14px;
             padding:48px;text-align:center;margin-top:8px'>
          <div style='font-size:32px;margin-bottom:12px'>⚙️</div>
          <div style='font-size:14px;font-weight:700;margin-bottom:6px'>Adjust parameters and click Run</div>
          <div style='font-family:Space Mono,monospace;font-size:10px;color:{MUTED}'>
            Changes are applied to the saved TF-IDF matrix — no re-preprocessing needed.
          </div>
        </div>
        """, unsafe_allow_html=True)