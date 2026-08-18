
import io
import html

import pandas as pd


# ============================================================
# CSV EXPORT
# ============================================================

def dataframe_to_csv_bytes(df):
    """Convert DataFrame to UTF-8 CSV bytes."""

    if df is None:
        return b""

    return df.to_csv(
        index=False
    ).encode("utf-8")


def dataframe_to_csv_string(df):
    """Convert DataFrame to CSV string."""

    if df is None:
        return ""

    return df.to_csv(
        index=False
    )


# ============================================================
# FORMATTING
# ============================================================

def format_number(value):
    """Format numbers for reports."""

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    if isinstance(value, (int, float)):
        return f"{value:,.2f}"

    return str(value)


def escape(value):
    """Safely escape text for HTML."""

    return html.escape(
        str(value)
    )


# ============================================================
# HTML HELPERS
# ============================================================

def metric_card(label, value, description=""):
    """Create a dashboard-style metric card."""

    return f"""
    <div class="metric-card">
        <div class="metric-label">
            {escape(label)}
        </div>

        <div class="metric-value">
            {escape(value)}
        </div>

        <div class="metric-description">
            {escape(description)}
        </div>
    </div>
    """


def section_header(title, subtitle=""):
    """Create report section heading."""

    subtitle_html = ""

    if subtitle:
        subtitle_html = f"""
        <div class="section-subtitle">
            {escape(subtitle)}
        </div>
        """

    return f"""
    <div class="section-header">
        <h2>{escape(title)}</h2>
        {subtitle_html}
    </div>
    """


def dataframe_to_html(
    dataframe,
    max_rows=20
):
    """Convert DataFrame to styled HTML table."""

    if dataframe is None:
        return """
        <div class="empty-state">
            No data available.
        </div>
        """

    if not isinstance(
        dataframe,
        pd.DataFrame
    ):
        return f"""
        <div class="text-box">
            {escape(dataframe)}
        </div>
        """

    if dataframe.empty:
        return """
        <div class="empty-state">
            No records available.
        </div>
        """

    display_df = dataframe.head(
        max_rows
    ).copy()

    return (
        display_df
        .to_html(
            index=False,
            classes="data-table",
            border=0,
            escape=True,
        )
    )


def bullet_list(items):
    """Create HTML bullet list."""

    if not items:
        return """
        <div class="empty-state">
            No findings available.
        </div>
        """

    return (
        "<ul class='bullet-list'>"
        +
        "".join(
            f"<li>{escape(item)}</li>"
            for item in items
        )
        +
        "</ul>"
    )


# ============================================================
# DATASET PROFILE
# ============================================================

def generate_profile_section(
    dataset_profile
):
    """Generate dataset overview section."""

    if not dataset_profile:
        return ""

    rows = dataset_profile.get(
        "rows",
        0
    )

    columns = dataset_profile.get(
        "columns",
        0
    )

    total_cells = dataset_profile.get(
        "total_cells",
        0
    )

    missing_cells = dataset_profile.get(
        "missing_cells",
        0
    )

    missing_percentage = dataset_profile.get(
        "missing_percentage",
        0
    )

    duplicate_rows = dataset_profile.get(
        "duplicate_rows",
        0
    )

    memory = dataset_profile.get(
        "memory_usage_mb",
        0
    )

    numerical = dataset_profile.get(
        "numerical_columns",
        []
    )

    categorical = dataset_profile.get(
        "categorical_columns",
        []
    )

    datetime_columns = dataset_profile.get(
        "datetime_columns",
        []
    )

    boolean_columns = dataset_profile.get(
        "boolean_columns",
        []
    )

    metrics_html = f"""
    <div class="metric-grid">

        {metric_card(
            "Rows",
            f"{rows:,}",
            "Total observations"
        )}

        {metric_card(
            "Columns",
            f"{columns:,}",
            "Dataset variables"
        )}

        {metric_card(
            "Total Cells",
            f"{total_cells:,}",
            "Rows × columns"
        )}

        {metric_card(
            "Missing Cells",
            f"{missing_cells:,}",
            f"{missing_percentage}% of dataset"
        )}

        {metric_card(
            "Duplicate Rows",
            f"{duplicate_rows:,}",
            "Exact duplicate observations"
        )}

        {metric_card(
            "Memory Usage",
            f"{memory} MB",
            "Approximate dataframe memory"
        )}

    </div>
    """

    column_types = f"""
    <div class="two-column">

        <div class="info-card">

            <h3>Numerical Columns</h3>

            <p>
                {escape(", ".join(numerical) if numerical else "None")}
            </p>

        </div>

        <div class="info-card">

            <h3>Categorical Columns</h3>

            <p>
                {escape(", ".join(categorical) if categorical else "None")}
            </p>

        </div>

        <div class="info-card">

            <h3>Datetime Columns</h3>

            <p>
                {escape(", ".join(datetime_columns) if datetime_columns else "None")}
            </p>

        </div>

        <div class="info-card">

            <h3>Boolean Columns</h3>

            <p>
                {escape(", ".join(boolean_columns) if boolean_columns else "None")}
            </p>

        </div>

    </div>
    """

    return (
        section_header(
            "Dataset Overview",
            "Structure and composition of the analyzed dataset."
        )
        +
        metrics_html
        +
        column_types
    )


# ============================================================
# QUALITY REPORT
# ============================================================

def generate_quality_section(
    quality_report
):
    """Generate data quality section."""

    if not quality_report:
        return ""

    metrics = quality_report.get(
        "metrics",
        {}
    )

    overall = quality_report.get(
        "overall_score"
    )

    completeness = quality_report.get(
        "completeness_score"
    )

    uniqueness = quality_report.get(
        "uniqueness_score"
    )

    duplicate = quality_report.get(
        "duplicate_score"
    )

    consistency = quality_report.get(
        "consistency_score"
    )

    outlier_rate = quality_report.get(
        "outlier_rate"
    )

    cards = f"""
    <div class="metric-grid">

        {metric_card(
            "Overall Quality",
            f"{format_number(overall)}/100",
            "Overall dataset quality"
        )}

        {metric_card(
            "Completeness",
            f"{format_number(completeness)}/100",
            "Missing-value completeness"
        )}

        {metric_card(
            "Uniqueness",
            f"{format_number(uniqueness)}/100",
            "Value uniqueness"
        )}

        {metric_card(
            "Consistency",
            f"{format_number(consistency)}/100",
            "Data consistency"
        )}

        {metric_card(
            "Duplicate Score",
            f"{format_number(duplicate)}/100",
            "Duplicate quality"
        )}

        {metric_card(
            "Outlier Rate",
            f"{format_number(outlier_rate)}%",
            "Detected outlier proportion"
        )}

    </div>
    """

    quality_details = f"""
    <div class="info-card">

        <h3>Quality Indicators</h3>

        <div class="detail-grid">

            <div>
                <span>Rows</span>
                <strong>
                    {metrics.get("rows", "N/A"):,}
                </strong>
            </div>

            <div>
                <span>Columns</span>
                <strong>
                    {metrics.get("columns", "N/A"):,}
                </strong>
            </div>

            <div>
                <span>Missing Values</span>
                <strong>
                    {metrics.get("missing_cells", "N/A"):,}
                </strong>
            </div>

            <div>
                <span>Duplicate Rows</span>
                <strong>
                    {metrics.get("duplicate_rows", "N/A"):,}
                </strong>
            </div>

        </div>

    </div>
    """

    return (
        section_header(
            "Data Quality",
            "Assessment of completeness, uniqueness, consistency and anomalies."
        )
        +
        cards
        +
        quality_details
    )


# ============================================================
# CLEANING IMPACT
# ============================================================

def generate_cleaning_section(
    cleaning_impact
):
    """Generate cleaning impact section."""

    if not cleaning_impact:
        return ""

    rows_removed = cleaning_impact.get(
        "rows_removed",
        0
    )

    columns_removed = cleaning_impact.get(
        "columns_removed",
        0
    )

    missing_removed = cleaning_impact.get(
        "missing_values_removed",
        0
    )

    duplicates_removed = cleaning_impact.get(
        "duplicates_removed",
        0
    )

    original_quality = cleaning_impact.get(
        "original_quality"
    )

    cleaned_quality = cleaning_impact.get(
        "cleaned_quality"
    )

    improvement = cleaning_impact.get(
        "quality_improvement"
    )

    cards = f"""
    <div class="metric-grid">

        {metric_card(
            "Rows Removed",
            f"{rows_removed:,}",
            "Observations removed during cleaning"
        )}

        {metric_card(
            "Columns Removed",
            f"{columns_removed:,}",
            "Variables removed"
        )}

        {metric_card(
            "Missing Values Removed",
            f"{missing_removed:,}",
            "Missing entries addressed"
        )}

        {metric_card(
            "Duplicates Removed",
            f"{duplicates_removed:,}",
            "Duplicate observations removed"
        )}

        {metric_card(
            "Original Quality",
            f"{format_number(original_quality)}/100",
            "Before cleaning"
        )}

        {metric_card(
            "Cleaned Quality",
            f"{format_number(cleaned_quality)}/100",
            "After cleaning"
        )}

    </div>

    <div class="highlight-box">

        <strong>
            Quality Improvement
        </strong>

        <span>
            {format_number(improvement)} points
        </span>

    </div>
    """

    return (
        section_header(
            "Cleaning Impact",
            "Effect of the data-cleaning operations performed."
        )
        +
        cards
    )


# ============================================================
# EDA
# ============================================================

def generate_eda_section(
    eda_report
):
    """Generate exploratory data analysis section."""

    if not eda_report:
        return ""

    sections = []

    descriptive = eda_report.get(
        "descriptive_statistics"
    )

    if isinstance(
        descriptive,
        pd.DataFrame
    ):

        sections.append(
            f"""
            <div class="subsection">

                <h3>
                    Descriptive Statistics
                </h3>

                {dataframe_to_html(
                    descriptive,
                    max_rows=20
                )}

            </div>
            """
        )

    categorical = eda_report.get(
        "categorical_statistics"
    )

    if isinstance(
        categorical,
        pd.DataFrame
    ):

        sections.append(
            f"""
            <div class="subsection">

                <h3>
                    Categorical Statistics
                </h3>

                {dataframe_to_html(
                    categorical,
                    max_rows=20
                )}

            </div>
            """
        )

    correlations = eda_report.get(
        "correlation_pairs"
    )

    if isinstance(
        correlations,
        pd.DataFrame
    ):

        sections.append(
            f"""
            <div class="subsection">

                <h3>
                    Strong Correlations
                </h3>

                {dataframe_to_html(
                    correlations,
                    max_rows=20
                )}

            </div>
            """
        )

    distribution = eda_report.get(
        "distribution_summary"
    )

    if isinstance(
        distribution,
        pd.DataFrame
    ):

        sections.append(
            f"""
            <div class="subsection">

                <h3>
                    Distribution Summary
                </h3>

                {dataframe_to_html(
                    distribution,
                    max_rows=20
                )}

            </div>
            """
        )

    missing = eda_report.get(
        "missing_analysis"
    )

    if isinstance(
        missing,
        pd.DataFrame
    ):

        sections.append(
            f"""
            <div class="subsection">

                <h3>
                    Missing Value Analysis
                </h3>

                {dataframe_to_html(
                    missing,
                    max_rows=20
                )}

            </div>
            """
        )

    findings = eda_report.get(
        "findings",
        []
    )

    if findings:

        sections.append(
            f"""
            <div class="subsection">

                <h3>
                    Key Findings
                </h3>

                {bullet_list(findings)}

            </div>
            """
        )

    return (
        section_header(
            "Exploratory Data Analysis",
            "Statistical patterns and relationships identified in the dataset."
        )
        +
        "".join(sections)
    )


# ============================================================
# OUTLIERS
# ============================================================

def generate_outlier_section(
    outlier_summary
):
    """Generate outlier analysis section."""

    if outlier_summary is None:
        return ""

    if isinstance(
        outlier_summary,
        pd.DataFrame
    ):

        return (
            section_header(
                "Outlier Analysis",
                "Potentially unusual observations identified in the dataset."
            )
            +
            dataframe_to_html(
                outlier_summary,
                max_rows=30
            )
        )

    return (
        section_header(
            "Outlier Analysis",
            "Potentially unusual observations identified in the dataset."
        )
        +
        f"""
        <div class="text-box">
            {escape(outlier_summary)}
        </div>
        """
    )


# ============================================================
# AI INSIGHTS
# ============================================================

def generate_ai_section(
    ai_insights
):
    """Generate AI-powered analysis section."""

    if not ai_insights:
        return ""

    ai_text = escape(
        str(ai_insights)
    )

    # Convert simple markdown formatting
    ai_text = ai_text.replace(
        "\n",
        "<br>"
    )

    return (
        section_header(
            "AI-Powered Analysis",
            "Insights generated by DataLens AI using the supplied dataset information."
        )
        +
        f"""
        <div class="ai-box">
            {ai_text}
        </div>
        """
    )


# ============================================================
# COMPLETE HTML REPORT
# ============================================================

def generate_html_report(
    dataset_profile=None,
    eda_report=None,
    quality_report=None,
    cleaning_impact=None,
    outlier_summary=None,
    ai_insights=None,
    title="DataLens Data Analysis Report"
):
    """Generate a complete professional HTML report."""

    sections = []

    if dataset_profile:
        sections.append(
            generate_profile_section(
                dataset_profile
            )
        )

    if quality_report:
        sections.append(
            generate_quality_section(
                quality_report
            )
        )

    if cleaning_impact:
        sections.append(
            generate_cleaning_section(
                cleaning_impact
            )
        )

    if eda_report:
        sections.append(
            generate_eda_section(
                eda_report
            )
        )

    if outlier_summary is not None:
        sections.append(
            generate_outlier_section(
                outlier_summary
            )
        )

    if ai_insights:
        sections.append(
            generate_ai_section(
                ai_insights
            )
        )

    body = "\n".join(
        section
        for section in sections
        if section
    )

    html_report = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
    {escape(title)}
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f4f6fa;
    color: #1f2937;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    line-height: 1.6;
}}

.report {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 50px 35px 80px;
}}

.cover {{
    background:
        linear-gradient(
            135deg,
            #17122e,
            #2c215e
        );
    color: white;
    border-radius: 22px;
    padding: 48px;
    margin-bottom: 35px;
    box-shadow:
        0 18px 45px rgba(0,0,0,0.12);
}}

.brand {{
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    opacity: 0.75;
}}

.cover h1 {{
    margin: 18px 0 10px;
    font-size: 38px;
    line-height: 1.15;
}}

.cover p {{
    margin: 0;
    color: #d8d2f3;
    font-size: 16px;
}}

.section-header {{
    margin-top: 42px;
    margin-bottom: 18px;
}}

.section-header h2 {{
    margin: 0;
    font-size: 25px;
    color: #151923;
}}

.section-subtitle {{
    margin-top: 4px;
    color: #697386;
    font-size: 14px;
}}

.metric-grid {{
    display: grid;
    grid-template-columns:
        repeat(3, 1fr);
    gap: 15px;
    margin-bottom: 20px;
}}

.metric-card {{
    background: white;
    border: 1px solid #e5e8ef;
    border-radius: 14px;
    padding: 20px;
    box-shadow:
        0 5px 18px rgba(20,25,40,0.04);
}}

.metric-label {{
    color: #7a8496;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.metric-value {{
    margin-top: 7px;
    font-size: 25px;
    font-weight: 800;
    color: #171b26;
}}

.metric-description {{
    margin-top: 4px;
    color: #8992a3;
    font-size: 12px;
}}

.two-column {{
    display: grid;
    grid-template-columns:
        repeat(2, 1fr);
    gap: 15px;
}}

.info-card {{
    background: white;
    border: 1px solid #e5e8ef;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}}

.info-card h3 {{
    margin-top: 0;
    font-size: 16px;
    color: #202532;
}}

.info-card p {{
    margin-bottom: 0;
    color: #657084;
    font-size: 14px;
    word-break: break-word;
}}

.detail-grid {{
    display: grid;
    grid-template-columns:
        repeat(2, 1fr);
    gap: 15px;
}}

.detail-grid div {{
    padding: 13px;
    background: #f7f8fb;
    border-radius: 10px;
}}

.detail-grid span {{
    display: block;
    color: #7c8799;
    font-size: 12px;
}}

.detail-grid strong {{
    display: block;
    margin-top: 2px;
    color: #202532;
    font-size: 16px;
}}

.highlight-box {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #eeebff;
    border: 1px solid #dcd5ff;
    border-radius: 14px;
    padding: 18px 20px;
    color: #4635a5;
}}

.highlight-box span {{
    font-size: 20px;
    font-weight: 800;
}}

.subsection {{
    background: white;
    border: 1px solid #e5e8ef;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
    overflow-x: auto;
}}

.subsection h3 {{
    margin-top: 0;
    color: #202532;
    font-size: 17px;
}}

.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

.data-table th {{
    background: #f1f3f7;
    color: #414a5a;
    font-weight: 700;
    text-align: left;
    padding: 10px;
    border-bottom: 1px solid #dfe3ea;
}}

.data-table td {{
    padding: 9px 10px;
    border-bottom: 1px solid #edf0f4;
    color: #5c6677;
}}

.bullet-list {{
    padding-left: 22px;
    color: #566174;
}}

.bullet-list li {{
    margin-bottom: 8px;
}}

.text-box {{
    background: white;
    border: 1px solid #e5e8ef;
    border-radius: 14px;
    padding: 20px;
    color: #596477;
    white-space: pre-wrap;
}}

.ai-box {{
    background:
        linear-gradient(
            135deg,
            #f2efff,
            #f7f5ff
        );
    border: 1px solid #ddd7ff;
    border-radius: 16px;
    padding: 25px;
    color: #3f3a5c;
    font-size: 14px;
    line-height: 1.8;
}}

.empty-state {{
    padding: 18px;
    background: #fafbfc;
    border-radius: 10px;
    color: #8a93a2;
    text-align: center;
}}

.footer {{
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #dfe3ea;
    color: #8a93a2;
    font-size: 12px;
    text-align: center;
}}

@media(max-width: 750px) {{

    .report {{
        padding: 25px 15px;
    }}

    .cover {{
        padding: 30px;
    }}

    .cover h1 {{
        font-size: 30px;
    }}

    .metric-grid {{
        grid-template-columns:
            1fr;
    }}

    .two-column {{
        grid-template-columns:
            1fr;
    }}

}}

@media print {{

    body {{
        background: white;
    }}

    .report {{
        max-width: none;
    }}

    .cover {{
        box-shadow: none;
    }}

    .metric-card,
    .info-card,
    .subsection {{
        break-inside: avoid;
    }}

}}

</style>

</head>

<body>

<div class="report">

    <div class="cover">

        <div class="brand">
            ◈ DataLens
        </div>

        <h1>
            {escape(title)}
        </h1>

        <p>
            Professional dataset analysis,
            data-quality assessment and
            actionable insights.
        </p>

    </div>

    {body}

    <div class="footer">
        Generated by DataLens ·
        AI-Powered Data Analysis Studio
    </div>

</div>

</body>

</html>
"""

    return html_report


# ============================================================
# TEXT REPORT
# ============================================================

def generate_text_report(
    dataset_profile=None,
    eda_report=None,
    quality_report=None,
    cleaning_impact=None,
    outlier_summary=None,
    ai_insights=None,
    title="Data Analysis Report"
):
    """
    Keep a text export available as a fallback.
    """

    sections = []

    sections.append(
        title
        + "\n"
        + "=" * len(title)
    )

    if dataset_profile:
        sections.append(
            generate_profile_section(
                dataset_profile
            )
        )

    if quality_report:
        sections.append(
            generate_quality_section(
                quality_report
            )
        )

    if cleaning_impact:
        sections.append(
            generate_cleaning_section(
                cleaning_impact
            )
        )

    if eda_report:
        sections.append(
            generate_eda_section(
                eda_report
            )
        )

    if outlier_summary is not None:
        sections.append(
            generate_outlier_section(
                outlier_summary
            )
        )

    if ai_insights:
        sections.append(
            generate_ai_section(
                ai_insights
            )
        )

    return "\n\n".join(
        section
        for section in sections
        if section
    )


def report_to_bytes(
    report_text
):
    """Convert text report to bytes."""

    return report_text.encode(
        "utf-8"
    )


def html_report_to_bytes(
    html_report
):
    """Convert HTML report to bytes."""

    return html_report.encode(
        "utf-8"
    )


# ============================================================
# REPORT BUNDLE
# ============================================================

def create_report_bundle(
    original_df,
    cleaned_df,
    dataset_profile=None,
    eda_report=None,
    quality_report=None,
    cleaning_impact=None,
    outlier_summary=None,
    ai_insights=None
):
    """Create all downloadable report assets."""

    report_text = generate_text_report(
        dataset_profile=dataset_profile,
        eda_report=eda_report,
        quality_report=quality_report,
        cleaning_impact=cleaning_impact,
        outlier_summary=outlier_summary,
        ai_insights=ai_insights
    )

    html_report = generate_html_report(
        dataset_profile=dataset_profile,
        eda_report=eda_report,
        quality_report=quality_report,
        cleaning_impact=cleaning_impact,
        outlier_summary=outlier_summary,
        ai_insights=ai_insights
    )

    return {

        "report_text":
            report_text,

        "report_bytes":
            report_to_bytes(
                report_text
            ),

        "html_report":
            html_report,

        "html_bytes":
            html_report_to_bytes(
                html_report
            ),

        "cleaned_csv":
            dataframe_to_csv_bytes(
                cleaned_df
            ),

        "original_csv":
            dataframe_to_csv_bytes(
                original_df
            )
    }