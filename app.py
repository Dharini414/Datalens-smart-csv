# ============================================================
# DATALENS
# ============================================================

import inspect
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DataLens | Data Analysis Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT MODULES
# ============================================================

from modules import ai_insights
from modules import cleaning
from modules import eda
from modules import metrics
from modules import outliers
from modules import profiler
from modules import quality
from modules import report
from modules import utils
from modules import visualization


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "cleaning_history" not in st.session_state:
    st.session_state.cleaning_history = []

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "column_ai_result" not in st.session_state:
    st.session_state.column_ai_result = None


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

/* =========================================================
   GLOBAL
   ========================================================= */

html,
body,
[class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 8% 0%,
            rgba(124, 92, 255, 0.13),
            transparent 26%
        ),
        radial-gradient(
            circle at 95% 8%,
            rgba(31, 199, 181, 0.07),
            transparent 24%
        ),
        #080b12;
}

.block-container {
    max-width: 1480px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================================================
   STREAMLIT CLEANUP
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* =========================================================
   TYPOGRAPHY
   ========================================================= */

h1 {
    font-size: 46px !important;
    line-height: 1.12 !important;
    letter-spacing: -0.045em !important;
    color: #f7f9fc !important;
}

h2 {
    font-size: 32px !important;
    line-height: 1.2 !important;
    letter-spacing: -0.035em !important;
    color: #f7f9fc !important;
}

h3 {
    font-size: 24px !important;
    line-height: 1.3 !important;
    color: #f1f4f9 !important;
}

h4 {
    font-size: 20px !important;
    color: #edf1f7 !important;
}

p,
.stMarkdown p {
    font-size: 17px;
    line-height: 1.7;
    color: #aeb8c8;
}

label,
.stSelectbox label,
.stMultiSelect label,
.stTextInput label,
.stTextArea label,
.stFileUploader label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #cdd5e1 !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #0a0e16;
    border-right: 1px solid rgba(255,255,255,0.075);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.25rem;
}

section[data-testid="stSidebar"] * {
    color: #e7ebf2;
}

.sidebar-brand {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.055em;
    color: #ffffff;
}

.sidebar-subtitle {
    margin-top: 2px;
    color: #788396;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button,
.stDownloadButton > button {
    min-height: 44px;
    border-radius: 11px;
    font-size: 16px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.09);
    background: #141a24;
    color: #edf2f8;
    transition: all 0.15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(124,92,255,0.60);
    color: #ffffff;
    background: #181f2c;
}

button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #7c5cff,
        #6045dc
    ) !important;

    border: none !important;
    color: white !important;
}


/* =========================================================
   INPUTS
   ========================================================= */

input,
textarea {
    background: #111722 !important;
    color: #f5f7fb !important;
    font-size: 16px !important;
}

[data-baseweb="select"] > div {
    background: #111722 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    min-height: 44px;
}

[data-baseweb="select"] span {
    color: #f1f4f8 !important;
    font-size: 16px !important;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background: #101620;
    border: 1px dashed rgba(124,92,255,0.42);
    border-radius: 13px;
    padding: 5px;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}


/* =========================================================
   METRICS
   ========================================================= */

[data-testid="stMetric"] {
    background: #101620;
    border: 1px solid rgba(255,255,255,0.075);
    border-radius: 14px;
    padding: 18px 18px 16px;
    min-height: 108px;
}

[data-testid="stMetricLabel"] {
    color: #8f9aac !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #f5f7fb !important;
    font-size: 32px !important;
    font-weight: 750 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 14px !important;
}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {
    color: #8f9aac !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}


/* =========================================================
   EXPANDERS
   ========================================================= */

[data-testid="stExpander"] {
    background: #101620;
    border: 1px solid rgba(255,255,255,0.075);
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    font-size: 16px !important;
    font-weight: 600 !important;
}


/* =========================================================
   HOME DASHBOARD CLASSES
   ========================================================= */

.dashboard-kicker {
    color: #8b78ff;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 9px;
}

.dashboard-title {
    color: #f8fafc;
    font-size: 46px;
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.055em;
}

.dashboard-description {
    color: #9ba6b7;
    font-size: 18px;
    line-height: 1.7;
    max-width: 820px;
    margin-top: 12px;
}

.dataset-name {
    color: #ffffff;
    font-size: 26px;
    font-weight: 750;
    letter-spacing: -0.035em;
}

.dataset-subtitle {
    color: #8995a7;
    font-size: 16px;
    margin-top: 4px;
}

.section-label {
    color: #79869a;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 12px;
    margin-bottom: 10px;
}

.info-text {
    color: #9aa6b7;
    font-size: 16px;
    line-height: 1.65;
}

.small-text {
    color: #7e8a9c;
    font-size: 14px;
}

</style>
""",
    unsafe_allow_html=True,
)
# ============================================================
# SAFE FUNCTION CALLER
# ============================================================

def safe_call(function, *args, **kwargs):
    """
    Calls an existing project function while tolerating
    optional/different parameter names.
    """

    try:

        signature = inspect.signature(function)
        parameters = signature.parameters

        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parameters.values()
        )

        if accepts_kwargs:
            valid_kwargs = kwargs
        else:
            valid_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in parameters
            }

        return function(
            *args,
            **valid_kwargs,
        )

    except Exception:
        return None


# ============================================================
# DATA HELPERS
# ============================================================

def numeric_columns(dataframe):
    return dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()


def categorical_columns(dataframe):
    return dataframe.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
        ]
    ).columns.tolist()


def dataset_stats(dataframe):

    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "numeric": len(
            numeric_columns(dataframe)
        ),
        "categorical": len(
            categorical_columns(dataframe)
        ),
        "missing": int(
            dataframe.isna().sum().sum()
        ),
        "duplicates": int(
            dataframe.duplicated().sum()
        ),
        "memory": round(
            dataframe.memory_usage(
                deep=True
            ).sum()
            / 1024
            / 1024,
            2,
        ),
    }


def style_plot(fig, height=500):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter",
            "color": "#dce3ed",
            "size": 13,
        },
        title={
            "font": {
                "size": 19,
                "color": "#f4f6fa",
            }
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 65,
            "b": 25,
        },
        height=height,
        hoverlabel={
            "bgcolor": "#151c27",
            "font_color": "#ffffff",
        },
        legend={
            "font": {
                "color": "#cbd4df"
            }
        },
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.055)",
        zerolinecolor="rgba(255,255,255,0.07)",
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.055)",
        zerolinecolor="rgba(255,255,255,0.07)",
    )

    return fig


def show_result(result):

    if result is None:
        return

    if isinstance(result, pd.DataFrame):

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True,
        )

    elif isinstance(result, pd.Series):

        st.dataframe(
            result.to_frame(),
            use_container_width=True,
        )

    elif isinstance(result, dict):

        simple_rows = []

        for key, value in result.items():

            if isinstance(
                value,
                (
                    dict,
                    list,
                    tuple,
                    pd.DataFrame,
                    pd.Series,
                ),
            ):
                continue

            simple_rows.append(
                {
                    "Metric": str(key)
                    .replace(
                        "_",
                        " ",
                    )
                    .title(),
                    "Value": value,
                }
            )

        if simple_rows:

            st.dataframe(
                pd.DataFrame(
                    simple_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.json(result)

    elif isinstance(
        result,
        (list, tuple),
    ):

        try:

            st.dataframe(
                pd.DataFrame(result),
                use_container_width=True,
                hide_index=True,
            )

        except Exception:

            st.write(result)

    else:

        st.write(result)


# ============================================================
# SIDEBAR — DATASET
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">◈ DataLens</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">Data Analysis Studio</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.caption("ANALYZE")

    analyze_pages = [
        "Overview",
        "Data Explorer",
        "EDA",
        "Visualization",
    ]

    for page in analyze_pages:

        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            ),
        ):

            st.session_state.page = page
            st.rerun()

    st.markdown("")

    st.caption("IMPROVE")

    improve_pages = [
        "Data Quality",
        "Cleaning",
        "Outliers",
    ]

    for page in improve_pages:

        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            ),
        ):

            st.session_state.page = page
            st.rerun()

    st.markdown("")

    st.caption("INSIGHTS")

    insight_pages = [
        "Metrics",
        "AI Insights",
        "Reports",
    ]

    for page in insight_pages:

        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            ),
        ):

            st.session_state.page = page
            st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # DATASET STATUS
    # --------------------------------------------------------

    st.caption("DATASET")

    if st.session_state.df is not None:

        sidebar_stats = dataset_stats(
            st.session_state.df
        )

        st.markdown(
            f"**{st.session_state.dataset_name}**"
        )

        st.caption(
            f"{sidebar_stats['rows']:,} rows · "
            f"{sidebar_stats['columns']:,} columns"
        )

        if st.button(
            "Reset dataset",
            use_container_width=True,
        ):

            st.session_state.df = (
                st.session_state.original_df.copy()
            )

            st.session_state.cleaning_history = []

            st.session_state.ai_result = None
            st.session_state.column_ai_result = None

            st.rerun()
        # --------------------------------------------------------
        # DOWNLOAD UPDATED DATASET
        # --------------------------------------------------------

        updated_csv = (
            st.session_state.df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="⬇ Download Updated Dataset",
            data=updated_csv,
            file_name="datalens_updated_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_updated_dataset",
        )
    

    else:

        st.caption(
            "No dataset loaded"
        )

        st.markdown(
            """
            <div style="
                margin-top:8px;
                padding:12px;
                border-radius:10px;
                background:rgba(255,255,255,0.025);
                border:1px solid rgba(255,255,255,0.06);
                color:#8f9aac;
                font-size:12px;
                line-height:1.55;
            ">
                Upload a CSV from the home page to unlock
                the full analysis workspace.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LANDING PAGE
# ============================================================

if st.session_state.df is None:

    # --------------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------------

    st.markdown(
        """
        <style>
        .landing-eyebrow {
            color: #9b8cff;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 14px;
        }

        .landing-heading {
            color: #f8fafc;
            font-size: 52px;
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: -0.055em;
            margin-bottom: 16px;
        }

        .landing-heading span {
            color: #8b78ff;
        }

        .landing-description {
            color: #a8b2c2;
            font-size: 17px;
            line-height: 1.75;
            max-width: 760px;
            margin-bottom: 8px;
        }

        .feature-title {
            color: #f4f6fa;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .feature-text {
            color: #929daf;
            font-size: 13px;
            line-height: 1.6;
        }

        .workflow-title {
            color: #f4f6fa;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .workflow-text {
            color: #929daf;
            font-size: 13px;
            line-height: 1.55;
        }

        .upload-label {
            color: #dce3ed;
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 7px;
        }

        .upload-description {
            color: #8e99aa;
            font-size: 13px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="landing-eyebrow">◈ PROFESSIONAL DATA ANALYSIS STUDIO</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="landing-heading">
            Turn raw data into<br>
            <span>clear, actionable insights.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="landing-description">
            DataLens gives you one intelligent workspace to explore,
            clean, visualize and understand your datasets — without
            jumping between multiple tools.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------
    # UPLOAD + HOW IT WORKS
    # --------------------------------------------------------

    upload_col, workflow_col = st.columns(
        [1.35, 1],
        gap="large",
    )

    with upload_col:

        st.markdown(
            '<div class="upload-label">Start with your dataset</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="upload-description">'
            'Upload a CSV and DataLens will prepare your analysis workspace.'
            '</div>',
            unsafe_allow_html=True,
        )

        uploaded_landing = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="landing_csv_upload",
            label_visibility="collapsed",
        )

        if uploaded_landing is not None:

            if (
                st.session_state.dataset_name
                != uploaded_landing.name
            ):

                try:

                    uploaded_landing.seek(0)

                    loaded = safe_call(
                        utils.read_csv_safely,
                        uploaded_landing,
                    )

                    if loaded is None:

                        uploaded_landing.seek(0)

                        loaded = pd.read_csv(
                            uploaded_landing
                        )

                    loaded = loaded.copy()

                    st.session_state.df = loaded.copy()

                    st.session_state.original_df = loaded.copy()

                    st.session_state.dataset_name = (
                        uploaded_landing.name
                    )

                    st.session_state.cleaning_history = []

                    st.session_state.ai_result = None

                    st.session_state.column_ai_result = None

                    st.session_state.page = "Overview"

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Unable to read CSV: {error}"
                    )

    with workflow_col:

        st.markdown(
            '<div class="workflow-title">How DataLens works</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="workflow-text">'
            'Upload → Profile → Explore → Improve → '
            'Visualize → Understand → Report'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        workflow = [
            ("01", "Profile", "Automatically understand structure, types and dataset health."),
            ("02", "Explore", "Investigate statistics, distributions and relationships."),
            ("03", "Improve", "Find missing values, duplicates, inconsistencies and outliers."),
            ("04", "Understand", "Use interactive charts and AI-powered analysis."),
        ]

        for number, title, description in workflow:

            c1, c2 = st.columns(
                [0.18, 0.82],
                gap="small",
            )

            with c1:
                st.markdown(
                    f"**{number}**"
                )

            with c2:
                st.markdown(
                    f"**{title}**  \n"
                    f"{description}"
                )

            st.markdown("")


    # --------------------------------------------------------
    # VALUE PROPOSITION
    # --------------------------------------------------------

    st.markdown("")

    st.markdown(
        "### Everything you need for a complete data workflow"
    )

    st.markdown(
        "From your first upload to your final report, "
        "DataLens brings the essential analysis tools together."
    )

    st.markdown("")

    feature_cols = st.columns(
        3,
        gap="medium",
    )

    features = [
        (
            "◉",
            "Explore & Profile",
            "Inspect dataset structure, column types, statistics, "
            "missing values, uniqueness and data characteristics."
        ),
        (
            "⌁",
            "Interactive EDA",
            "Understand distributions, categorical patterns, "
            "correlations and relationships through interactive analysis."
        ),
        (
            "◈",
            "Visualization Studio",
            "Create histograms, scatter plots, box plots, bar charts, "
            "line charts, pie charts, area charts and heatmaps."
        ),
        (
            "✓",
            "Data Quality",
            "Measure completeness, consistency, uniqueness, "
            "duplicates and overall dataset quality."
        ),
        (
            "◆",
            "Smart Cleaning",
            "Handle missing values, duplicates, empty rows, "
            "empty columns and data-type conversions."
        ),
        (
            "✦",
            "AI-Powered Insights",
            "Ask questions about your dataset and receive "
            "AI-generated analytical observations and recommendations."
        ),
    ]

    for index, (
        icon,
        title,
        description,
    ) in enumerate(features):

        with feature_cols[index % 3]:

            st.markdown(
                f"### {icon} {title}"
            )

            st.markdown(
                description
            )

            if index % 3 == 2:
                st.markdown("")


    # --------------------------------------------------------
    # ANALYSIS CAPABILITIES
    # --------------------------------------------------------

    st.markdown("")

    st.markdown(
        "### A complete analytical workspace"
    )

    capability_cols = st.columns(
        4,
        gap="medium",
    )

    capabilities = [
        (
            "10+",
            "Analysis modules",
            "EDA, quality, cleaning, profiling, metrics and more."
        ),
        (
            "9",
            "Interactive charts",
            "Explore your data visually with Plotly-powered charts."
        ),
        (
            "3",
            "Outlier methods",
            "IQR, Z-Score and Isolation Forest detection."
        ),
        (
            "1",
            "Unified workspace",
            "Move from raw CSV to analysis and reporting in one place."
        ),
    ]

    for column, (
        number,
        title,
        description,
    ) in zip(
        capability_cols,
        capabilities,
    ):

        with column:

            st.metric(
                title,
                number,
            )

            st.caption(
                description
            )


    # --------------------------------------------------------
    # FINAL CTA
    # --------------------------------------------------------

    st.markdown("")

    st.markdown(
        "### Ready to explore your data?"
    )

    st.markdown(
        "Upload a CSV above and let DataLens turn your dataset "
        "into an interactive analytical workspace."
    )

    st.markdown("")

    st.stop()



# ============================================================
# CURRENT DATASET
# ============================================================

df = st.session_state.df
stats = dataset_stats(df)


# ============================================================
# OVERVIEW / HOME
# ============================================================

if st.session_state.page == "Overview":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.markdown(
        '<div class="dashboard-kicker">DATASET DASHBOARD</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-title">'
        'Your data, at a glance.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-description">'
        'Explore your dataset, identify quality issues, '
        'visualize patterns and move directly into deeper analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.markdown(
        '<div class="dataset-name">'
        f'{st.session_state.dataset_name}'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dataset-subtitle">'
        f'{stats["rows"]:,} records · '
        f'{stats["columns"]:,} columns · '
        f'{stats["numeric"]:,} numeric · '
        f'{stats["categorical"]:,} categorical'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------
    # KEY METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Rows",
            f"{stats['rows']:,}",
        )

    with c2:
        st.metric(
            "Columns",
            f"{stats['columns']:,}",
        )

    with c3:
        st.metric(
            "Missing values",
            f"{stats['missing']:,}",
        )

    with c4:
        st.metric(
            "Duplicates",
            f"{stats['duplicates']:,}",
        )

    with c5:
        st.metric(
            "Memory",
            f"{stats['memory']} MB",
        )

    st.markdown("")

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">QUICK ACTIONS</div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        if st.button(
            "Explore dataset",
            use_container_width=True,
        ):

            st.session_state.page = "Data Explorer"
            st.rerun()

    with q2:

        if st.button(
            "Check data quality",
            use_container_width=True,
        ):

            st.session_state.page = "Data Quality"
            st.rerun()

    with q3:

        if st.button(
            "Detect outliers",
            use_container_width=True,
        ):

            st.session_state.page = "Outliers"
            st.rerun()

    with q4:

        if st.button(
            "Generate AI insights",
            use_container_width=True,
        ):

            st.session_state.page = "AI Insights"
            st.rerun()

    st.markdown("")

    # --------------------------------------------------------
    # INTERACTIVE EXPLORATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">INTERACTIVE EXPLORATION</div>',
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns(
        [1, 2],
        gap="large",
    )

    with chart_left:

        st.subheader(
            "Choose what to explore"
        )

        numeric_cols = numeric_columns(df)
        categorical_cols = categorical_columns(df)

        analysis_type = st.radio(
            "Analysis",
            [
                "Distribution",
                "Scatter relationship",
                "Category breakdown",
                "Missing values",
            ],
        )

        if analysis_type == "Distribution":

            if numeric_cols:

                selected_column = st.selectbox(
                    "Numeric column",
                    numeric_cols,
                    key="home_distribution_column",
                )

            else:

                selected_column = None
                st.warning(
                    "No numeric columns found."
                )

        elif analysis_type == "Scatter relationship":

            if len(numeric_cols) >= 2:

                x_column = st.selectbox(
                    "X axis",
                    numeric_cols,
                    key="home_scatter_x",
                )

                y_candidates = [
                    col
                    for col in numeric_cols
                    if col != x_column
                ]

                y_column = st.selectbox(
                    "Y axis",
                    y_candidates,
                    key="home_scatter_y",
                )

            else:

                x_column = None
                y_column = None

                st.warning(
                    "At least two numeric columns are required."
                )

        elif analysis_type == "Category breakdown":

            if categorical_cols:

                selected_category = st.selectbox(
                    "Category",
                    categorical_cols,
                    key="home_category",
                )

            else:

                selected_category = None

                st.warning(
                    "No categorical columns found."
                )

    with chart_right:

        if analysis_type == "Distribution":

            if numeric_cols:

                fig = px.histogram(
                    df,
                    x=selected_column,
                    marginal="box",
                    nbins=35,
                    title=f"Distribution · {selected_column}",
                )

                style_plot(
                    fig,
                    height=500,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                )

        elif analysis_type == "Scatter relationship":

            if (
                x_column is not None
                and y_column is not None
            ):

                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    trendline="ols",
                    title=(
                        f"{x_column} vs {y_column}"
                    ),
                )

                style_plot(
                    fig,
                    height=500,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                )

        elif analysis_type == "Category breakdown":

            if selected_category is not None:

                counts = (
                    df[selected_category]
                    .astype(str)
                    .value_counts()
                    .head(15)
                    .sort_values(
                        ascending=True
                    )
                )

                fig = px.bar(
                    x=counts.values,
                    y=counts.index,
                    orientation="h",
                    title=(
                        f"Top categories · "
                        f"{selected_category}"
                    ),
                )

                style_plot(
                    fig,
                    height=500,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                )

        elif analysis_type == "Missing values":

            missing = (
                df.isna()
                .sum()
                .sort_values(
                    ascending=True
                )
            )

            missing = missing[
                missing > 0
            ]

            if missing.empty:

                st.success(
                    "Excellent — no missing values were detected."
                )

            else:

                fig = px.bar(
                    x=missing.values,
                    y=missing.index,
                    orientation="h",
                    title="Missing values by column",
                    labels={
                        "x": "Missing values",
                        "y": "Column",
                    },
                )

                style_plot(
                    fig,
                    height=500,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                )

    # --------------------------------------------------------
    # DATA QUALITY SNAPSHOT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">DATA HEALTH</div>',
        unsafe_allow_html=True,
    )

    health_left, health_right = st.columns(
        [1, 1],
        gap="large",
    )

    with health_left:

        st.subheader(
            "Quality snapshot"
        )

        completeness = (
            100
            * (
                1
                - (
                    df.isna().sum().sum()
                    / max(
                        df.size,
                        1,
                    )
                )
            )
        )

        duplicate_rate = (
            df.duplicated().mean()
            * 100
        )

        qc1, qc2 = st.columns(2)

        with qc1:

            st.metric(
                "Completeness",
                f"{completeness:.1f}%",
            )

        with qc2:

            st.metric(
                "Duplicate rate",
                f"{duplicate_rate:.1f}%",
            )

        if completeness >= 95:

            st.success(
                "The dataset has strong completeness."
            )

        elif completeness >= 80:

            st.warning(
                "Some missing-value treatment may be useful."
            )

        else:

            st.error(
                "The dataset contains substantial missing data."
            )

    with health_right:

        st.subheader(
            "Column composition"
        )

        composition = pd.DataFrame(
            {
                "Type": [
                    "Numeric",
                    "Categorical",
                    "Other",
                ],
                "Columns": [
                    stats["numeric"],
                    stats["categorical"],
                    max(
                        0,
                        stats["columns"]
                        - stats["numeric"]
                        - stats["categorical"],
                    ),
                ],
            }
        )

        fig = px.pie(
            composition,
            names="Type",
            values="Columns",
            hole=0.58,
            title="Column composition",
        )

        style_plot(
            fig,
            height=360,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
            },
        )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">RELATIONSHIPS</div>',
        unsafe_allow_html=True,
    )

    nums = numeric_columns(df)

    if len(nums) >= 2:

        with st.expander(
            "Open correlation explorer",
            expanded=False,
        ):

            corr = df[nums].corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Numeric feature correlation",
            )

            style_plot(
                fig,
                height=600,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                },
            )

    else:

        st.info(
            "Correlation analysis requires at least two numeric columns."
        )

    # --------------------------------------------------------
    # QUICK EXPORTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">QUICK EXPORT</div>',
        unsafe_allow_html=True,
    )

    e1, e2, e3 = st.columns(3)

    with e1:

        csv_bytes = df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "Download dataset",
            data=csv_bytes,
            file_name="datalens_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with e2:

        if st.button(
            "Open reports",
            use_container_width=True,
        ):

            st.session_state.page = "Reports"
            st.rerun()

    with e3:

        if st.button(
            "Open full EDA",
            use_container_width=True,
        ):

            st.session_state.page = "EDA"
            st.rerun()


# ============================================================
# DATA EXPLORER
# ============================================================

elif st.session_state.page == "Data Explorer":

    st.title("Data Explorer")

    st.write(
        "Inspect records and understand the structure of every column."
    )

    search = st.text_input(
        "Search columns",
        placeholder="Search by column name...",
    )

    rows = st.select_slider(
        "Rows",
        options=[
            10,
            25,
            50,
            100,
            250,
        ],
        value=25,
    )

    visible_columns = list(
        df.columns
    )

    if search:

        visible_columns = [
            col
            for col in visible_columns
            if search.lower()
            in str(col).lower()
        ]

    if visible_columns:

        st.dataframe(
            df[visible_columns].head(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "Column profile"
    )

    profile_rows = []

    for column in df.columns:

        series = df[column]

        profile_rows.append(
            {
                "Column": column,
                "Type": str(series.dtype),
                "Non-null": int(
                    series.notna().sum()
                ),
                "Missing": int(
                    series.isna().sum()
                ),
                "Unique": int(
                    series.nunique(
                        dropna=True
                    )
                ),
                "Missing %": round(
                    series.isna().mean()
                    * 100,
                    2,
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(profile_rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# EDA
# ============================================================

elif st.session_state.page == "EDA":

    st.title("Exploratory Data Analysis")

    st.write(
        "Investigate distributions, descriptive statistics, categories and relationships."
    )

    tabs = st.tabs(
        [
            "Statistics",
            "Distributions",
            "Categories",
            "Correlation",
        ]
    )

    with tabs[0]:

        result = safe_call(
            eda.get_descriptive_statistics,
            df,
        )

        if result is not None:

            show_result(result)

        else:

            st.dataframe(
                df.describe(
                    include="all"
                ).transpose(),
                use_container_width=True,
            )

    with tabs[1]:

        nums = numeric_columns(df)

        if nums:

            column = st.selectbox(
                "Numeric column",
                nums,
                key="eda_distribution_column",
            )

            result = safe_call(
                eda.get_distribution_summary,
                df,
                column=column,
            )

            if result is not None:
                show_result(result)

            fig = px.histogram(
                df,
                x=column,
                marginal="box",
                nbins=40,
                title=f"Distribution · {column}",
            )

            style_plot(fig)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No numeric columns available."
            )

    with tabs[2]:

        cats = categorical_columns(df)

        if cats:

            column = st.selectbox(
                "Categorical column",
                cats,
                key="eda_category_column",
            )

            result = safe_call(
                eda.get_categorical_summary,
                df,
                column=column,
            )

            if result is not None:
                show_result(result)

            counts = (
                df[column]
                .astype(str)
                .value_counts()
                .head(20)
            )

            fig = px.bar(
                x=counts.index,
                y=counts.values,
                title=f"Categories · {column}",
            )

            style_plot(fig)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No categorical columns available."
            )

    with tabs[3]:

        nums = numeric_columns(df)

        if len(nums) >= 2:

            corr = safe_call(
                eda.get_correlation_matrix,
                df,
            )

            if not isinstance(
                corr,
                pd.DataFrame,
            ):

                corr = df[nums].corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Matrix",
            )

            style_plot(fig)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            strong = safe_call(
                eda.get_strong_correlations,
                df,
            )

            if strong is not None:

                st.subheader(
                    "Strong correlations"
                )

                show_result(strong)

        else:

            st.info(
                "At least two numeric columns are required."
            )


# ============================================================
# DATA QUALITY
# ============================================================

elif st.session_state.page == "Data Quality":

    st.title("Data Quality")

    st.write(
        "Measure completeness, consistency, uniqueness and overall data health."
    )

    quality_score = safe_call(
        quality.calculate_quality_score,
        df,
    )

    completeness = safe_call(
        quality.calculate_completeness_score,
        df,
    )

    uniqueness = safe_call(
        quality.calculate_uniqueness_score,
        df,
    )

    consistency = safe_call(
        quality.calculate_consistency_score,
        df,
    )

    outlier_score = safe_call(
        quality.calculate_outlier_score,
        df,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    values = [
        quality_score,
        completeness,
        uniqueness,
        consistency,
        outlier_score,
    ]

    labels = [
        "Quality",
        "Completeness",
        "Uniqueness",
        "Consistency",
        "Outlier score",
    ]

    for col, label, value in zip(
        [c1, c2, c3, c4, c5],
        labels,
        values,
    ):

        with col:

            if isinstance(
                value,
                (int, float, np.number),
            ):

                st.metric(
                    label,
                    f"{float(value):.1f}",
                )

            else:

                st.metric(
                    label,
                    "—",
                )

    st.subheader(
        "Quality report"
    )

    result = safe_call(
        quality.get_quality_report,
        df,
    )

    if result is not None:

        show_result(result)

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    st.subheader(
        "Missing-value analysis"
    )

    if missing.empty:

        st.success(
            "No missing values detected."
        )

    else:

        missing_df = pd.DataFrame(
            {
                "Column": missing.index,
                "Missing": missing.values,
                "Missing %": (
                    missing.values
                    / len(df)
                    * 100
                ).round(2),
            }
        )

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CLEANING
# ============================================================

elif st.session_state.page == "Cleaning":

    st.title("Data Cleaning")

    st.write(
        "Analyze and clean the working dataset while "
        "preserving the original upload."
    )

    st.info(
        "All changes affect the current working dataset. "
        "Use Reset dataset in the sidebar to restore the original."
    )

    # ========================================================
    # CLEANING ANALYSIS
    # ========================================================

    st.markdown("---")

    st.subheader("🔎 Cleaning Analysis")

    total_rows = len(df)

    total_columns = len(df.columns)

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing_total = int(
        df.isna().sum().sum()
    )

    missing_columns = [
        col
        for col in df.columns
        if df[col].isna().any()
    ]

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    # --------------------------------------------------------
    # EMPTY ROWS / COLUMNS
    # --------------------------------------------------------

    empty_rows = int(
        df.isna().all(axis=1).sum()
    )

    empty_columns = int(
        df.isna().all(axis=0).sum()
    )

    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Missing Values",
            f"{missing_total:,}"
        )

    with c2:

        st.metric(
            "Duplicate Rows",
            f"{duplicate_count:,}"
        )

    with c3:

        st.metric(
            "Empty Rows",
            f"{empty_rows:,}"
        )

    with c4:

        st.metric(
            "Empty Columns",
            f"{empty_columns:,}"
        )

    # ========================================================
    # MISSING VALUE ANALYSIS
    # ========================================================

    st.markdown("---")

    st.subheader("🧩 Missing Value Analysis")

    if missing_total == 0:

        st.success(
            "✓ No missing values detected in the dataset."
        )

    else:

        missing_data = []

        for col in df.columns:

            count = int(
                df[col].isna().sum()
            )

            if count > 0:

                missing_data.append(
                    {
                        "Column": col,
                        "Missing Values": count,
                        "Missing %": round(
                            count / total_rows * 100,
                            2
                        ),
                    }
                )

        missing_table = (
            pd.DataFrame(
                missing_data
            )
            .sort_values(
                "Missing Values",
                ascending=False
            )
        )

        st.warning(
            f"⚠ {missing_total:,} missing values "
            f"found across "
            f"{len(missing_columns)} column(s)."
        )

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Review the missing-value distribution before "
            "choosing a treatment method."
        )

        # ----------------------------------------------------
        # REMOVE ALL MISSING ROWS
        # ----------------------------------------------------

        if st.button(
            "🗑 Remove Rows With Missing Values",
            type="primary",
            use_container_width=True,
            key="analysis_remove_missing_rows",
        ):

            result = safe_call(
                cleaning.drop_rows_with_missing_values,
                df,
            )

            if isinstance(
                result,
                pd.DataFrame,
            ):

                st.session_state.df = result

            else:

                st.session_state.df = (
                    df.dropna()
                )

            st.session_state.cleaning_history.append(
                "Removed rows containing missing values"
            )

            st.success(
                "Rows containing missing values were removed."
            )

            st.rerun()

    # ========================================================
    # DUPLICATE ANALYSIS
    # ========================================================

    st.markdown("---")

    st.subheader("📋 Duplicate Analysis")

    if duplicate_count == 0:

        st.success(
            "✓ No duplicate rows detected."
        )

    else:

        duplicate_percentage = round(
            duplicate_count
            / total_rows
            * 100,
            2
        )

        st.warning(
            f"⚠ {duplicate_count:,} duplicate rows detected "
            f"({duplicate_percentage}% of the dataset)."
        )

        # ----------------------------------------------------
        # SHOW DUPLICATES
        # ----------------------------------------------------

        duplicate_rows = safe_call(
            cleaning.get_duplicate_rows,
            df,
        )

        if isinstance(
            duplicate_rows,
            pd.DataFrame,
        ) and not duplicate_rows.empty:

            with st.expander(
                "View duplicate rows"
            ):

                st.dataframe(
                    duplicate_rows,
                    use_container_width=True,
                    hide_index=True,
                )

        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        if st.button(
            "🗑 Remove Duplicate Rows",
            type="primary",
            use_container_width=True,
            key="analysis_remove_duplicates",
        ):

            result = safe_call(
                cleaning.remove_duplicates,
                df,
            )

            if isinstance(
                result,
                pd.DataFrame,
            ):

                st.session_state.df = result

            else:

                st.session_state.df = (
                    df.drop_duplicates()
                )

            st.session_state.cleaning_history.append(
                "Removed duplicate rows"
            )

            st.success(
                "Duplicate rows were removed."
            )

            st.rerun()

    # ========================================================
    # EMPTY DATA ANALYSIS
    # ========================================================

    st.markdown("---")

    st.subheader("🧹 Empty Data Analysis")

    empty_c1, empty_c2 = st.columns(2)

    with empty_c1:

        st.metric(
            "Completely Empty Rows",
            f"{empty_rows:,}"
        )

    with empty_c2:

        st.metric(
            "Completely Empty Columns",
            f"{empty_columns:,}"
        )

    # --------------------------------------------------------
    # EMPTY ROWS
    # --------------------------------------------------------

    if empty_rows > 0:

        st.warning(
            f"⚠ {empty_rows:,} completely empty row(s) detected."
        )

        if st.button(
            "🗑 Remove Empty Rows",
            use_container_width=True,
            key="analysis_remove_empty_rows",
        ):

            result = safe_call(
                cleaning.remove_empty_rows,
                df,
            )

            st.session_state.df = (
                result
                if isinstance(
                    result,
                    pd.DataFrame,
                )
                else df.dropna(
                    how="all"
                )
            )

            st.session_state.cleaning_history.append(
                "Removed empty rows"
            )

            st.rerun()

    # --------------------------------------------------------
    # EMPTY COLUMNS
    # --------------------------------------------------------

    if empty_columns > 0:

        st.warning(
            f"⚠ {empty_columns:,} completely empty "
            f"column(s) detected."
        )

        if st.button(
            "🗑 Remove Empty Columns",
            use_container_width=True,
            key="analysis_remove_empty_columns",
        ):

            result = safe_call(
                cleaning.remove_empty_columns,
                df,
            )

            st.session_state.df = (
                result
                if isinstance(
                    result,
                    pd.DataFrame,
                )
                else df.dropna(
                    axis=1,
                    how="all",
                )
            )

            st.session_state.cleaning_history.append(
                "Removed empty columns"
            )

            st.rerun()

    if (
        missing_total == 0
        and duplicate_count == 0
        and empty_rows == 0
        and empty_columns == 0
    ):

        st.success(
            "✨ No major cleaning issues detected. "
            "Your dataset currently looks clean."
        )

    # ========================================================
    # CLEANING OPERATIONS
    # ========================================================

    st.markdown("---")

    st.subheader("🛠 Cleaning Operations")

    tabs = st.tabs(
        [
            "Missing Values",
            "Duplicates",
            "Columns",
        ]
    )

    # ========================================================
    # MISSING VALUES TAB
    # ========================================================

    with tabs[0]:

        missing_columns = [
            col
            for col in df.columns
            if df[col].isna().any()
        ]

        if not missing_columns:

            st.success(
                "No missing values found."
            )

        else:

            column = st.selectbox(
                "Column",
                missing_columns,
                key="clean_missing_column",
            )

            strategy = st.selectbox(
                "Treatment",
                [
                    "Mean",
                    "Median",
                    "Mode",
                    "Forward Fill",
                    "Backward Fill",
                    "Drop Rows",
                ],
                key="clean_missing_strategy",
            )

            if st.button(
                "Apply treatment",
                type="primary",
                key="apply_missing_treatment",
            ):

                strategy_value = {
                    "Mean": "mean",
                    "Median": "median",
                    "Mode": "mode",
                    "Forward Fill": "ffill",
                    "Backward Fill": "bfill",
                    "Drop Rows": "drop",
                }[strategy]

                result = safe_call(
                    cleaning.fill_missing_values,
                    df,
                    column=column,
                    strategy=strategy_value,
                )

                if isinstance(
                    result,
                    pd.DataFrame,
                ):

                    st.session_state.df = result

                else:

                    working = df.copy()

                    if strategy == "Mean":

                        if pd.api.types.is_numeric_dtype(
                            working[column]
                        ):

                            working[column] = (
                                working[column]
                                .fillna(
                                    working[column].mean()
                                )
                            )

                    elif strategy == "Median":

                        if pd.api.types.is_numeric_dtype(
                            working[column]
                        ):

                            working[column] = (
                                working[column]
                                .fillna(
                                    working[column].median()
                                )
                            )

                    elif strategy == "Mode":

                        mode = working[
                            column
                        ].mode()

                        if not mode.empty:

                            working[column] = (
                                working[column]
                                .fillna(
                                    mode.iloc[0]
                                )
                            )

                    elif strategy == "Forward Fill":

                        working[column] = (
                            working[column]
                            .ffill()
                        )

                    elif strategy == "Backward Fill":

                        working[column] = (
                            working[column]
                            .bfill()
                        )

                    else:

                        working = (
                            working
                            .dropna(
                                subset=[column]
                            )
                        )

                    st.session_state.df = (
                        working
                    )

                st.session_state.cleaning_history.append(
                    f"{strategy} → {column}"
                )

                st.success(
                    "Cleaning operation completed."
                )

                st.rerun()

    # ========================================================
    # DUPLICATES TAB
    # ========================================================

    with tabs[1]:

        duplicate_count = int(
            df.duplicated().sum()
        )

        st.metric(
            "Duplicate rows",
            f"{duplicate_count:,}",
        )

        if duplicate_count:

            if st.button(
                "Remove duplicates",
                type="primary",
                key="remove_duplicates_tab",
            ):

                result = safe_call(
                    cleaning.remove_duplicates,
                    df,
                )

                if isinstance(
                    result,
                    pd.DataFrame,
                ):

                    st.session_state.df = result

                else:

                    st.session_state.df = (
                        df.drop_duplicates()
                    )

                st.session_state.cleaning_history.append(
                    "Removed duplicate rows"
                )

                st.rerun()

        else:

            st.success(
                "No duplicate rows detected."
            )

        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "Remove empty rows",
                use_container_width=True,
                key="remove_empty_rows_tab",
            ):

                result = safe_call(
                    cleaning.remove_empty_rows,
                    df,
                )

                st.session_state.df = (
                    result
                    if isinstance(
                        result,
                        pd.DataFrame,
                    )
                    else df.dropna(
                        how="all"
                    )
                )

                st.session_state.cleaning_history.append(
                    "Removed empty rows"
                )

                st.rerun()

        with c2:

            if st.button(
                "Remove empty columns",
                use_container_width=True,
                key="remove_empty_columns_tab",
            ):

                result = safe_call(
                    cleaning.remove_empty_columns,
                    df,
                )

                st.session_state.df = (
                    result
                    if isinstance(
                        result,
                        pd.DataFrame,
                    )
                    else df.dropna(
                        axis=1,
                        how="all",
                    )
                )

                st.session_state.cleaning_history.append(
                    "Removed empty columns"
                )

                st.rerun()

    # ========================================================
    # COLUMNS TAB
    # ========================================================

    with tabs[2]:

        column = st.selectbox(
            "Column",
            df.columns.tolist(),
            key="conversion_column",
        )

        dtype = st.selectbox(
            "Convert to",
            [
                "int",
                "float",
                "str",
                "bool",
                "datetime",
            ],
            key="conversion_dtype",
        )

        if st.button(
            "Convert column",
            type="primary",
            key="convert_column_button",
        ):

            result = safe_call(
                cleaning.convert_column_type,
                df,
                column=column,
                dtype=dtype,
            )

            if isinstance(
                result,
                pd.DataFrame,
            ):

                st.session_state.df = result

            else:

                working = df.copy()

                try:

                    if dtype == "datetime":

                        working[column] = (
                            pd.to_datetime(
                                working[column],
                                errors="coerce",
                            )
                        )

                    else:

                        working[column] = (
                            working[column]
                            .astype(dtype)
                        )

                    st.session_state.df = (
                        working
                    )

                except Exception as error:

                    st.error(
                        str(error)
                    )

            st.session_state.cleaning_history.append(
                f"Converted {column} → {dtype}"
            )

            st.rerun()

        st.markdown("---")

        remove_column = st.selectbox(
            "Remove column",
            df.columns.tolist(),
            key="remove_column",
        )

        if st.button(
            "Remove selected column",
            key="remove_selected_column",
        ):

            result = safe_call(
                cleaning.drop_columns,
                df,
                columns=[remove_column],
            )

            st.session_state.df = (
                result
                if isinstance(
                    result,
                    pd.DataFrame,
                )
                else df.drop(
                    columns=[remove_column]
                )
            )

            st.session_state.cleaning_history.append(
                f"Removed column → {remove_column}"
            )

            st.rerun()

    # ========================================================
    # CLEANING HISTORY
    # ========================================================

    if st.session_state.cleaning_history:

        st.markdown("---")

        st.subheader(
            "Cleaning History"
        )

        history = pd.DataFrame(
            {
                "Step": range(
                    1,
                    len(
                        st.session_state.cleaning_history
                    ) + 1,
                ),
                "Operation": (
                    st.session_state.cleaning_history
                ),
            }
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# OUTLIERS
# ============================================================

elif st.session_state.page == "Outliers":

    st.title("Outlier Detection")

    st.write(
        "Identify unusual observations using statistical "
        "and machine-learning methods."
    )

    nums = numeric_columns(df)

    if not nums:

        st.warning(
            "No numeric columns are available."
        )

    else:

        # ====================================================
        # OUTLIER ANALYSIS
        # ====================================================

        st.markdown("---")

        st.subheader("🔎 Outlier Analysis")

        st.write(
            "DataLens analyzes numerical columns using the "
            "IQR method to identify observations that fall "
            "outside the expected range."
        )

        # ----------------------------------------------------
        # ANALYZE ALL NUMERICAL COLUMNS
        # ----------------------------------------------------

        outlier_data = []

        for col in nums:

            try:

                series = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if series.empty:
                    continue

                q1 = series.quantile(
                    0.25
                )

                q3 = series.quantile(
                    0.75
                )

                iqr_value = q3 - q1

                lower = (
                    q1
                    - 1.5 * iqr_value
                )

                upper = (
                    q3
                    + 1.5 * iqr_value
                )

                mask = (
                    (series < lower)
                    |
                    (series > upper)
                )

                count = int(
                    mask.sum()
                )

                percentage = round(
                    count
                    / len(series)
                    * 100,
                    2
                )

                outlier_data.append(
                    {
                        "Column": col,
                        "Outliers": count,
                        "Outlier %": percentage,
                        "Lower Bound": round(
                            lower,
                            3
                        ),
                        "Upper Bound": round(
                            upper,
                            3
                        ),
                    }
                )

            except Exception:
                continue

        if outlier_data:

            outlier_table = pd.DataFrame(
                outlier_data
            )

            total_outliers = int(
                outlier_table[
                    "Outliers"
                ].sum()
            )

            columns_with_outliers = int(
                (
                    outlier_table[
                        "Outliers"
                    ] > 0
                ).sum()
            )

        else:

            outlier_table = pd.DataFrame()

            total_outliers = 0

            columns_with_outliers = 0

        # ====================================================
        # OUTLIER SUMMARY METRICS
        # ====================================================

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Numeric Columns",
                f"{len(nums):,}"
            )

        with c2:

            st.metric(
                "Columns With Outliers",
                f"{columns_with_outliers:,}"
            )

        with c3:

            st.metric(
                "Potential Outliers",
                f"{total_outliers:,}"
            )

        # ====================================================
        # OUTLIER RESULT
        # ====================================================

        if total_outliers == 0:

            st.success(
                "✓ No potential IQR outliers were detected "
                "in the numerical columns."
            )

        else:

            st.warning(
                f"⚠ {total_outliers:,} potential outlier "
                f"observations detected across "
                f"{columns_with_outliers:,} column(s)."
            )

            st.dataframe(
                outlier_table[
                    outlier_table[
                        "Outliers"
                    ] > 0
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                "Outliers are not automatically errors. "
                "They may represent legitimate extreme "
                "observations. Review them before removal."
            )

            # ------------------------------------------------
            # REMOVE OUTLIERS
            # ------------------------------------------------

            if st.button(
                "🗑 Remove Detected Outlier Rows",
                type="primary",
                use_container_width=True,
                key="remove_detected_outliers",
            ):

                working = df.copy()

                original_rows = len(
                    working
                )

                for col in nums:

                    series = pd.to_numeric(
                        working[col],
                        errors="coerce"
                    )

                    q1 = series.quantile(
                        0.25
                    )

                    q3 = series.quantile(
                        0.75
                    )

                    iqr_value = (
                        q3 - q1
                    )

                    lower = (
                        q1
                        - 1.5
                        * iqr_value
                    )

                    upper = (
                        q3
                        + 1.5
                        * iqr_value
                    )

                    working = working[
                        series.isna()
                        |
                        series.between(
                            lower,
                            upper
                        )
                    ]

                removed_rows = (
                    original_rows
                    - len(working)
                )

                st.session_state.df = (
                    working
                )

                st.session_state.cleaning_history.append(
                    f"Removed {removed_rows:,} "
                    "outlier rows using IQR"
                )

                st.success(
                    f"Removed {removed_rows:,} "
                    "outlier rows."
                )

                st.rerun()

        # ====================================================
        # DETAILED OUTLIER DETECTION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Detailed Outlier Detection"
        )

        method = st.selectbox(
            "Detection method",
            [
                "IQR",
                "Z-Score",
                "Isolation Forest",
            ],
            key="outlier_method",
        )

        column = st.selectbox(
            "Column",
            nums,
            key="outlier_column",
        )

        if method == "IQR":

            result = safe_call(
                outliers.detect_iqr_outliers,
                df,
                column=column,
            )

        elif method == "Z-Score":

            result = safe_call(
                outliers.detect_zscore_outliers,
                df,
                column=column,
            )

        else:

            result = safe_call(
                outliers.detect_isolation_forest_outliers,
                df,
                columns=[column],
            )

        if result is not None:

            st.subheader(
                f"{method} Results · {column}"
            )

            show_result(
                result
            )

        # ====================================================
        # BOXPLOT
        # ====================================================

        fig = px.box(
            df,
            y=column,
            points="outliers",
            title=(
                f"Outlier Distribution · {column}"
            ),
        )

        style_plot(
            fig
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ====================================================
        # EXISTING OUTLIER SUMMARY
        # ====================================================

        summary = safe_call(
            outliers.get_outlier_summary,
            df,
        )

        if summary is not None:

            st.subheader(
                "Dataset Outlier Summary"
            )

            show_result(
                summary
            )
# ============================================================
# VISUALIZATION
# ============================================================

elif st.session_state.page == "Visualization":

    st.title("Visualization Studio")

    st.write(
        "Build interactive charts from your dataset."
    )

    chart_type = st.selectbox(
        "Chart type",
        [
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "Bar Chart",
            "Line Chart",
            "Area Chart",
            "Pie Chart",
            "Violin Plot",
            "Correlation Heatmap",
        ],
    )

    x = st.selectbox(
        "X / Category",
        df.columns.tolist(),
    )

    y = st.selectbox(
        "Y / Value",
        ["None"] + df.columns.tolist(),
    )

    fig = None

    try:

        if chart_type == "Histogram":

            fig = px.histogram(
                df,
                x=x,
                marginal="box",
                nbins=40,
            )

        elif chart_type == "Box Plot":

            fig = px.box(
                df,
                y=x,
                points="outliers",
            )

        elif chart_type == "Scatter Plot":

            if y != "None":

                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    trendline="ols",
                )

        elif chart_type == "Bar Chart":

            counts = (
                df[x]
                .astype(str)
                .value_counts()
                .head(20)
            )

            fig = px.bar(
                x=counts.index,
                y=counts.values,
            )

        elif chart_type == "Line Chart":

            if y != "None":

                fig = px.line(
                    df,
                    x=x,
                    y=y,
                )

        elif chart_type == "Area Chart":

            if y != "None":

                fig = px.area(
                    df,
                    x=x,
                    y=y,
                )

        elif chart_type == "Pie Chart":

            counts = (
                df[x]
                .astype(str)
                .value_counts()
                .head(15)
            )

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.45,
            )

        elif chart_type == "Violin Plot":

            fig = px.violin(
                df,
                y=x,
                box=True,
                points="outliers",
            )

        elif chart_type == "Correlation Heatmap":

            nums = numeric_columns(df)

            if len(nums) >= 2:

                fig = px.imshow(
                    df[nums].corr(),
                    text_auto=".2f",
                    aspect="auto",
                )

    except Exception as error:

        st.error(
            f"Could not create visualization: {error}"
        )

    if fig is not None:

        style_plot(
            fig,
            height=580,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

    else:

        st.info(
            "Select the required columns to generate this visualization."
        )


# ============================================================
# METRICS
# ============================================================

elif st.session_state.page == "Metrics":

    st.title("Dataset Metrics")

    st.write(
        "Quantitative measurements calculated from your dataset."
    )

    result = safe_call(
        metrics.calculate_dataset_metrics,
        df,
    )

    if result is not None:

        show_result(result)

    st.subheader(
        "Column metrics"
    )

    column = st.selectbox(
        "Column",
        df.columns.tolist(),
    )

    result = safe_call(
        metrics.get_column_metrics,
        df,
        column=column,
    )

    if result is not None:

        show_result(result)

    nums = numeric_columns(df)

    if nums:

        st.subheader(
            "Numeric metrics"
        )

        result = safe_call(
            metrics.get_numeric_metrics,
            df,
        )

        if result is not None:

            show_result(result)


# ============================================================
# AI INSIGHTS
# ============================================================

elif st.session_state.page == "AI Insights":

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if "ai_question_input" not in st.session_state:
        st.session_state.ai_question_input = ""

    if "ai_result" not in st.session_state:
        st.session_state.ai_result = None

    if "column_ai_result" not in st.session_state:
        st.session_state.column_ai_result = None

    # --------------------------------------------------------
    # QUICK QUESTION CALLBACK
    # --------------------------------------------------------

    def select_quick_question(question_text):

        # This callback runs BEFORE the text_area is
        # instantiated on the next Streamlit rerun.
        #
        # Therefore it is safe to update the widget state here.

        st.session_state.ai_question_input = question_text

        # Remove old AI answer
        st.session_state.ai_result = None

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="dashboard-kicker">AI DATA ANALYST</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-title">'
        'Turn your dataset into actionable insights.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-description">'
        'Ask questions about your dataset and let Gemini '
        'analyze patterns, data quality, relationships, '
        'anomalies and recommended next steps.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ========================================================
    # DATASET CHECK
    # ========================================================

    if df is None:

        st.warning(
            "Please upload a dataset first to use DataLens AI."
        )

    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    st.subheader("⚡ Quick Questions")

    st.write(
        "Choose a question below to automatically fill "
        "the AI question box."
    )

    quick_questions = {

        "📊 Dataset Overview":
            "Give me a concise overview of this dataset, "
            "including its size, columns, data types, "
            "and the most important characteristics.",

        "🔍 Find Data Quality Issues":
            "Identify the most important data quality issues "
            "in this dataset, including missing values, "
            "duplicates, invalid values, and other potential problems.",

        "📈 Find Important Patterns":
            "Identify the most important numerical and categorical "
            "patterns in this dataset. Use actual values from "
            "the dataset where possible.",

        "🚨 Find Anomalies":
            "Identify potential anomalies or unusual values "
            "in this dataset. Explain why they may be unusual "
            "and mention the relevant columns.",

        "💡 Give Me Key Insights":
            "Give me the most important actionable insights "
            "from this dataset. Focus only on observations "
            "supported by the available data.",

        "🧹 What Should I Clean?":
            "Tell me what should be cleaned or improved "
            "in this dataset and explain the recommended "
            "cleaning steps."
    }

    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    cols = st.columns(3)

    for index, (button_label, question_text) in enumerate(
        quick_questions.items()
    ):

        with cols[index % 3]:

            st.button(
                button_label,
                key=f"quick_question_{index}",
                use_container_width=True,
                on_click=select_quick_question,
                args=(question_text,),
            )

    # ========================================================
    # QUESTION INPUT
    # ========================================================

    st.markdown("")

    st.subheader("Ask DataLens AI")

    question = st.text_area(
        "What would you like to investigate?",

        placeholder=(
            "Example:\n"
            "• What are the most important patterns in this dataset?\n"
            "• Are there any data quality problems?\n"
            "• Which variables appear to have meaningful relationships?\n"
            "• Give me actionable insights from this dataset."
        ),

        height=150,

        key="ai_question_input",
    )

    # ========================================================
    # GENERATE AI ANALYSIS
    # ========================================================

    generate_clicked = st.button(
        "✨ Generate AI Analysis",
        type="primary",
        use_container_width=True,
        key="generate_ai_analysis",
    )

    if generate_clicked:

        if df is None:

            st.error(
                "Please upload a dataset before asking DataLens AI."
            )

        else:

            final_question = question.strip()

            if not final_question:

                final_question = (
                    "Provide a complete professional analysis "
                    "of this dataset, including important patterns, "
                    "data quality issues, relationships, anomalies "
                    "and recommendations."
                )

            # Clear previous result
            st.session_state.ai_result = None

            # ------------------------------------------------
            # CALL GEMINI
            # ------------------------------------------------

            with st.spinner(
                "DataLens AI is analyzing your dataset..."
            ):

                try:

                    result = ai_insights.generate_data_insights(

                        df=df,

                        cleaned_df=(
                            cleaned_df
                            if "cleaned_df" in locals()
                            else None
                        ),

                        outlier_summary=(
                            outlier_summary
                            if "outlier_summary" in locals()
                            else None
                        ),

                        metrics=(
                            metrics
                            if "metrics" in locals()
                            else None
                        ),

                        question=final_question,
                    )

                    if result is None:

                        raise RuntimeError(
                            "The AI function returned no result."
                        )

                    result = str(result).strip()

                    if not result:

                        raise RuntimeError(
                            "Gemini returned an empty response."
                        )

                    st.session_state.ai_result = result

                except Exception as error:

                    st.session_state.ai_result = None

                    st.error(
                        "AI analysis could not be generated."
                    )

                    st.exception(error)

    # ========================================================
    # AI OUTPUT
    # ========================================================

    if st.session_state.get("ai_result"):

        st.markdown("---")

        st.markdown(
            '<div class="section-label">AI ANALYSIS</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "✨ DataLens AI Findings"
        )

        # Render Gemini Markdown directly
        st.markdown(
            st.session_state.ai_result
        )

    # ========================================================
    # COLUMN ANALYSIS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-label">COLUMN ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "Analyze a specific column"
    )

    st.write(
        "Get a focused interpretation of one column "
        "including its statistics, patterns and data quality."
    )

    if df is not None and len(df.columns) > 0:

        column = st.selectbox(
            "Select column",
            df.columns.tolist(),
            key="ai_column",
        )

        analyze_column_clicked = st.button(
            "🔎 Analyze Column",
            use_container_width=True,
            key="analyze_ai_column",
        )

        if analyze_column_clicked:

            st.session_state.column_ai_result = None

            with st.spinner(
                f"Analyzing '{column}'..."
            ):

                try:

                    result = ai_insights.generate_column_insight(
                        df,
                        column,
                    )

                    if result is None:

                        raise RuntimeError(
                            "The column insight function "
                            "returned no result."
                        )

                    result = str(result).strip()

                    if not result:

                        raise RuntimeError(
                            "Gemini returned an empty "
                            "column insight."
                        )

                    st.session_state.column_ai_result = result

                except Exception as error:

                    st.session_state.column_ai_result = None

                    st.error(
                        "Column analysis could not be generated."
                    )

                    st.exception(error)

    else:

        st.info(
            "Upload a dataset to enable column-level AI analysis."
        )

    # ========================================================
    # COLUMN OUTPUT
    # ========================================================

    if st.session_state.get("column_ai_result"):

        st.markdown("")

        st.markdown(
            '<div class="section-label">COLUMN INSIGHT</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "🤖 AI Column Analysis"
        )

        st.markdown(
            st.session_state.column_ai_result
        )
# ============================================================
# REPORTS
# ============================================================

elif st.session_state.page == "Reports":

    st.title("Analysis Reports")

    st.write(
        "Generate a professional report containing your dataset "
        "profile, quality assessment, cleaning impact, EDA, "
        "outlier analysis and AI-powered insights."
    )

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if st.session_state.df is None:

        st.warning(
            "Upload a dataset first to generate a report."
        )

    else:

        current_df = st.session_state.df
        original_df = st.session_state.original_df

        # ----------------------------------------------------
        # BUILD DATASET PROFILE
        # ----------------------------------------------------

        dataset_profile = safe_call(
            profiler.get_dataset_profile,
            current_df,
        )

        # ----------------------------------------------------
        # QUALITY REPORT
        # ----------------------------------------------------

        quality_report = safe_call(
            quality.get_quality_report,
            current_df,
        )

        # ----------------------------------------------------
        # EDA REPORT
        # ----------------------------------------------------

        eda_report = {}

        descriptive_statistics = safe_call(
            eda.get_descriptive_statistics,
            current_df,
        )

        categorical_statistics = safe_call(
            eda.get_categorical_summary,
            current_df,
        )

        distribution_summary = safe_call(
            eda.get_distribution_summary,
            current_df,
        )

        correlation_pairs = safe_call(
            eda.get_strong_correlations,
            current_df,
        )

        eda_report["descriptive_statistics"] = (
            descriptive_statistics
        )

        eda_report["categorical_statistics"] = (
            categorical_statistics
        )

        eda_report["distribution_summary"] = (
            distribution_summary
        )

        eda_report["correlation_pairs"] = (
            correlation_pairs
        )

        # ----------------------------------------------------
        # OUTLIER SUMMARY
        # ----------------------------------------------------

        outlier_summary = safe_call(
            outliers.get_outlier_summary,
            current_df,
        )

        # ----------------------------------------------------
        # CLEANING IMPACT
        # ----------------------------------------------------

        cleaning_impact = safe_call(
            quality.get_cleaning_impact,
            original_df,
            current_df,
        )

        # ----------------------------------------------------
        # AI INSIGHTS
        # ----------------------------------------------------

        ai_insights = st.session_state.get(
            "ai_result",
            None
        )

        # ----------------------------------------------------
        # CREATE REPORT BUNDLE
        # ----------------------------------------------------

        report_bundle = report.create_report_bundle(

            original_df=original_df,

            cleaned_df=current_df,

            dataset_profile=dataset_profile,

            eda_report=eda_report,

            quality_report=quality_report,

            cleaning_impact=cleaning_impact,

            outlier_summary=outlier_summary,

            ai_insights=ai_insights,
        )

        # ====================================================
        # REPORT PREVIEW
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📄 DataLens Analysis Report"
        )

        st.write(
            "Your report includes:"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.info(
                "📊 Dataset Profile\n\n"
                "Structure, size, column types and memory usage."
            )

        with c2:
            st.info(
                "🧹 Data Quality\n\n"
                "Missing values, duplicates, completeness and quality scores."
            )

        with c3:
            st.info(
                "📈 Analysis\n\n"
                "EDA, correlations, outliers and AI insights."
            )

        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Download Report"
        )

        st.download_button(

            label="📄 Download Professional HTML Report",

            data=report_bundle["html_bytes"],

            file_name=(
                f"{st.session_state.dataset_name}"
                "_DataLens_Report.html"
            ),

            mime="text/html",

            use_container_width=True,
        )

        # ----------------------------------------------------
        # DOWNLOAD DATASET
        # ----------------------------------------------------

        st.download_button(

            label="⬇️ Download Cleaned Dataset",

            data=report_bundle["cleaned_csv"],

            file_name=(
                f"{st.session_state.dataset_name}"
                "_cleaned.csv"
            ),

            mime="text/csv",

            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "DataLens · Professional Data Analysis Studio · "
    "Built with Python, Pandas, Plotly & Streamlit"
)