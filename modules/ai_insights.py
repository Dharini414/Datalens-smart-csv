# ============================================================
# DATALENS
# AI INSIGHTS MODULE
# Gemini-powered dataset and column analysis
# ============================================================

import os
import json

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-3.5-flash"


# ============================================================
# API KEY
# ============================================================

def _get_api_key():

    # --------------------------------------------------------
    # Streamlit Secrets
    # --------------------------------------------------------

    try:

        import streamlit as st

        if "GOOGLE_API_KEY" in st.secrets:

            api_key = st.secrets["GOOGLE_API_KEY"]

            if api_key:

                return str(api_key).strip()

    except Exception:
        pass

    # --------------------------------------------------------
    # Environment variable / .env
    # --------------------------------------------------------

    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:

        return api_key.strip()

    return None


# ============================================================
# GEMINI CLIENT
# ============================================================

def _get_client():

    api_key = _get_api_key()

    if not api_key:

        raise ValueError(
            "GOOGLE_API_KEY was not found.\n\n"
            "For local development, add this to your .env file:\n\n"
            "GOOGLE_API_KEY=your_api_key\n\n"
            "For Streamlit deployment, add GOOGLE_API_KEY "
            "to Streamlit Secrets."
        )

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception as error:

        raise RuntimeError(
            f"Could not initialize Gemini client: {error}"
        ) from error


# ============================================================
# SAFE JSON CONVERSION
# ============================================================

def _safe_json(data):

    try:

        return json.dumps(
            data,
            indent=2,
            default=str
        )

    except Exception:

        return str(data)


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def _extract_response_text(response):

    if response is None:

        raise RuntimeError(
            "Gemini returned no response."
        )

    # --------------------------------------------------------
    # Standard response.text
    # --------------------------------------------------------

    try:

        text = response.text

        if text:

            text = str(text).strip()

            if text:

                return text

    except Exception:
        pass

    # --------------------------------------------------------
    # Fallback: inspect candidates
    # --------------------------------------------------------

    try:

        candidates = getattr(
            response,
            "candidates",
            None
        )

        if candidates:

            parts = []

            for candidate in candidates:

                content = getattr(
                    candidate,
                    "content",
                    None
                )

                if content is None:
                    continue

                parts_list = getattr(
                    content,
                    "parts",
                    []
                )

                for part in parts_list:

                    text_part = getattr(
                        part,
                        "text",
                        None
                    )

                    if text_part:

                        parts.append(
                            str(text_part)
                        )

            if parts:

                text = "\n".join(parts).strip()

                if text:

                    return text

    except Exception:
        pass

    raise RuntimeError(
        "Gemini returned a response, "
        "but no text content was available."
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def _prepare_dataframe(
    df,
    max_rows=100
):

    if df is None:

        raise ValueError(
            "Dataframe cannot be None."
        )

    sample = df.head(
        max_rows
    ).copy()

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    for column in sample.columns:

        try:

            sample[column] = sample[column].replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # Convert datetime columns
    # --------------------------------------------------------

    for column in sample.columns:

        try:

            if pd.api.types.is_datetime64_any_dtype(
                sample[column]
            ):

                sample[column] = (
                    sample[column]
                    .astype(str)
                )

        except Exception:

            pass

    # --------------------------------------------------------
    # Convert problematic objects to strings
    # --------------------------------------------------------

    for column in sample.columns:

        try:

            if sample[column].dtype == "object":

                sample[column] = (
                    sample[column]
                    .apply(
                        lambda x:
                        str(x)
                        if pd.notna(x)
                        else None
                    )
                )

        except Exception:

            pass

    return sample


# ============================================================
# DATASET SUMMARY
# ============================================================

def _build_dataset_summary(df):

    if df is None:

        raise ValueError(
            "Dataframe is required."
        )

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )

    summary = {

        "rows": int(
            len(df)
        ),

        "columns": int(
            len(df.columns)
        ),

        "column_names": (
            df.columns.tolist()
        ),

        "data_types": {

            column: str(
                df[column].dtype
            )

            for column in df.columns
        },

        "missing_values": {

            column: int(
                df[column]
                .isna()
                .sum()
            )

            for column in df.columns
        },

        "missing_percentages": {

            column: round(
                float(
                    df[column]
                    .isna()
                    .mean()
                    * 100
                ),
                2
            )

            for column in df.columns
        },

        "unique_values": {

            column: int(
                df[column]
                .nunique(
                    dropna=True
                )
            )

            for column in df.columns
        },

        "duplicate_rows": int(
            df.duplicated()
            .sum()
        )
    }

    # ========================================================
    # NUMERIC SUMMARY
    # ========================================================

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    summary[
        "numeric_summary"
    ] = {}

    for column in numeric_columns:

        series = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )

        if series.empty:

            continue

        summary[
            "numeric_summary"
        ][column] = {

            "mean": round(
                float(
                    series.mean()
                ),
                4
            ),

            "median": round(
                float(
                    series.median()
                ),
                4
            ),

            "std": round(
                float(
                    series.std()
                ),
                4
            ),

            "min": round(
                float(
                    series.min()
                ),
                4
            ),

            "max": round(
                float(
                    series.max()
                ),
                4
            ),

            "skewness": round(
                float(
                    series.skew()
                ),
                4
            )
        }

    # ========================================================
    # CATEGORICAL SUMMARY
    # ========================================================

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    summary[
        "categorical_summary"
    ] = {}

    for column in categorical_columns:

        values = (
            df[column]
            .value_counts(
                dropna=False
            )
            .head(10)
            .to_dict()
        )

        summary[
            "categorical_summary"
        ][column] = {

            "top_values": {

                str(key): int(value)

                for key, value
                in values.items()
            }
        }

    return summary


# ============================================================
# CORRELATION SUMMARY
# ============================================================

def _build_correlation_summary(df):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
    )

    if numeric_df.shape[1] < 2:

        return {}

    try:

        correlation = (
            numeric_df
            .corr()
        )

        relationships = []

        columns = correlation.columns

        for i in range(
            len(columns)
        ):

            for j in range(
                i + 1,
                len(columns)
            ):

                value = correlation.iloc[
                    i,
                    j
                ]

                if pd.isna(value):

                    continue

                relationships.append({

                    "column_1": columns[i],

                    "column_2": columns[j],

                    "correlation": round(
                        float(value),
                        4
                    )
                })

        relationships.sort(
            key=lambda x:
            abs(
                x["correlation"]
            ),
            reverse=True
        )

        return relationships[:15]

    except Exception:

        return {}


# ============================================================
# MAIN AI DATA INSIGHTS
# ============================================================

def generate_data_insights(
    df,
    cleaned_df=None,
    outlier_summary=None,
    metrics=None,
    question=None
):

    if df is None:

        raise ValueError(
            "A dataframe is required."
        )

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:

        raise ValueError(
            "The uploaded dataset is empty."
        )

    # ========================================================
    # DATA SUMMARY
    # ========================================================

    summary = _build_dataset_summary(
        df
    )

    sample = _prepare_dataframe(
        df,
        max_rows=100
    )

    correlation_summary = (
        _build_correlation_summary(
            df
        )
    )

    # ========================================================
    # CLEANED DATASET
    # ========================================================

    cleaned_summary = None

    if cleaned_df is not None:

        try:

            cleaned_summary = (
                _build_dataset_summary(
                    cleaned_df
                )
            )

        except Exception:

            cleaned_summary = None

    # ========================================================
    # USER QUESTION
    # ========================================================

    if (
        isinstance(
            question,
            str
        )
        and question.strip()
    ):

        user_question = (
            question.strip()
        )

    else:

        user_question = (
            "Provide a complete professional "
            "analysis of this dataset."
        )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are DataLens AI, an experienced professional
data analyst.

Analyze the supplied dataset carefully.

USER QUESTION:
{user_question}

Answer the user's question directly and clearly.

Do not invent information.

Only make claims that are supported by the
provided dataset information.

============================================================
DATASET SUMMARY
============================================================

{_safe_json(summary)}

============================================================
NUMERICAL RELATIONSHIPS
============================================================

{_safe_json(correlation_summary)}

============================================================
SAMPLE ROWS
============================================================

{sample.to_json(
    orient="records",
    date_format="iso"
)}

============================================================
CLEANED DATASET SUMMARY
============================================================

{_safe_json(cleaned_summary)}

============================================================
OUTLIER INFORMATION
============================================================

{_safe_json(outlier_summary)}

============================================================
DATASET METRICS
============================================================

{_safe_json(metrics)}

============================================================
RESPONSE FORMAT
============================================================

## Dataset Overview

Explain the size, structure and general characteristics
of the dataset.

## Data Quality

Discuss missing values, duplicates and completeness.

## Important Numerical Patterns

Discuss meaningful numerical statistics, ranges,
distributions and skewness.

## Categorical Patterns

Discuss important categorical columns and dominant
categories when applicable.

## Relationships Between Variables

Discuss correlations or relationships only when
supported by the supplied information.

## Potential Anomalies

Identify unusual values or potential anomalies when
there is evidence.

## Cleaning Recommendations

Suggest practical cleaning actions.

## Key Business or Analytical Insights

Give concise and actionable observations.

## Recommended Next Steps

Suggest useful next analytical steps.

============================================================
RULES
============================================================

1. Never invent facts.

2. Mention actual column names.

3. Use actual numerical values where useful.

4. Clearly distinguish observations from recommendations.

5. If something cannot be determined, say so.

6. Do not treat the sample rows as the complete dataset.

7. Keep the answer professional and readable.

8. Use Markdown headings and bullet points.

9. Answer the user's specific question first.
"""

    # ========================================================
    # GEMINI
    # ========================================================

    client = _get_client()

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

    except Exception as error:

        raise RuntimeError(
            f"Gemini API request failed: {error}"
        ) from error

    # ========================================================
    # EXTRACT RESPONSE
    # ========================================================

    return _extract_response_text(
        response
    )


# ============================================================
# COLUMN AI INSIGHT
# ============================================================

def generate_column_insight(
    df,
    column
):

    if df is None:

        raise ValueError(
            "A dataframe is required."
        )

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if column not in df.columns:

        raise ValueError(
            f"Column '{column}' does not exist."
        )

    series = df[column]

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    information = {

        "column": column,

        "data_type": str(
            series.dtype
        ),

        "total_rows": int(
            len(series)
        ),

        "missing": int(
            series.isna().sum()
        ),

        "missing_percentage": round(
            float(
                series.isna().mean()
                * 100
            ),
            2
        ),

        "unique_values": int(
            series.nunique(
                dropna=True
            )
        )
    }

    # ========================================================
    # NUMERIC COLUMN
    # ========================================================

    if pd.api.types.is_numeric_dtype(
        series
    ):

        numeric = (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )

        if not numeric.empty:

            information[
                "statistics"
            ] = {

                "mean": round(
                    float(
                        numeric.mean()
                    ),
                    4
                ),

                "median": round(
                    float(
                        numeric.median()
                    ),
                    4
                ),

                "minimum": round(
                    float(
                        numeric.min()
                    ),
                    4
                ),

                "maximum": round(
                    float(
                        numeric.max()
                    ),
                    4
                ),

                "std": round(
                    float(
                        numeric.std()
                    ),
                    4
                ),

                "skewness": round(
                    float(
                        numeric.skew()
                    ),
                    4
                )
            }

    # ========================================================
    # CATEGORICAL COLUMN
    # ========================================================

    else:

        information[
            "top_values"
        ] = {

            str(key): int(value)

            for key, value in (
                series
                .value_counts(
                    dropna=False
                )
                .head(10)
                .items()
            )
        }

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are DataLens AI, a professional data analyst.

Analyze this dataset column using ONLY the information
provided below.

COLUMN INFORMATION:

{_safe_json(information)}

Provide a concise and useful analysis.

Use exactly these sections:

## Column Overview

Explain what can reasonably be inferred.

## Important Patterns

Identify meaningful patterns supported by the data.

## Data Quality

Discuss missing values, uniqueness and quality issues.

## Potential Anomalies

Mention unusual characteristics only when supported.

## Analytical Considerations

Explain how the column could be useful for further analysis.

IMPORTANT:

- Do not invent information.
- Do not assume the meaning beyond what the data supports.
- Mention actual statistics.
- Clearly distinguish observations from assumptions.
- Use Markdown.
"""

    # ========================================================
    # GEMINI
    # ========================================================

    client = _get_client()

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

    except Exception as error:

        raise RuntimeError(
            f"Gemini API request failed: {error}"
        ) from error

    # ========================================================
    # RETURN TEXT
    # ========================================================

    return _extract_response_text(
        response
    )


# ============================================================
# QUICK QUESTION HELPER
# ============================================================

def generate_quick_question(
    df,
    question,
    cleaned_df=None,
    outlier_summary=None,
    metrics=None
):

    """
    Used by DataLens Quick Questions.

    Example:

        answer = generate_quick_question(
            df,
            "What are the most important issues in this dataset?"
        )
    """

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    return generate_data_insights(
        df=df,
        cleaned_df=cleaned_df,
        outlier_summary=outlier_summary,
        metrics=metrics,
        question=question
    )