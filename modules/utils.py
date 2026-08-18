import io
import re
from datetime import datetime

import numpy as np
import pandas as pd


def validate_dataframe(df):
    """
    Validate that the uploaded object is
    a usable DataFrame.
    """

    if df is None:
        raise ValueError(
            "No dataset was provided."
        )

    if not isinstance(
        df,
        pd.DataFrame
    ):
        raise TypeError(
            "The uploaded data must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "The dataset contains no columns."
        )

    return True


def validate_csv_file(
    uploaded_file,
    max_size_mb=200
):
    """
    Validate an uploaded CSV file.

    Checks:
    - File exists
    - File extension
    - File size
    """

    if uploaded_file is None:
        raise ValueError(
            "Please upload a CSV file."
        )

    file_name = getattr(
        uploaded_file,
        "name",
        ""
    )

    if not file_name.lower().endswith(
        ".csv"
    ):
        raise ValueError(
            "Only CSV files are supported."
        )

    file_size = getattr(
        uploaded_file,
        "size",
        None
    )

    if file_size is not None:

        max_size = (
            max_size_mb
            * 1024
            * 1024
        )

        if file_size > max_size:

            raise ValueError(
                f"File size exceeds "
                f"{max_size_mb} MB."
            )

    return True


def read_csv_safely(
    uploaded_file
):
    """
    Read an uploaded CSV while handling
    common encoding issues.
    """

    validate_csv_file(
        uploaded_file
    )

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1",
        "cp1252"
    ]

    last_error = None

    for encoding in encodings:

        try:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding=encoding
            )

            validate_dataframe(
                df
            )

            return df

        except Exception as error:

            last_error = error

    raise ValueError(
        "Unable to read the CSV file. "
        f"Last error: {last_error}"
    )


def clean_column_names(df):
    """
    Standardize column names.

    Example:

    'Customer Name'
        -> 'Customer_Name'

    'Sales ($)'
        -> 'Sales'
    """

    data = df.copy()

    cleaned_names = []

    for column in data.columns:

        name = str(column).strip()

        name = re.sub(
            r"[^\w\s-]",
            "",
            name
        )

        name = re.sub(
            r"[\s-]+",
            "_",
            name
        )

        name = name.strip("_")

        if not name:

            name = "column"

        cleaned_names.append(
            name
        )

    cleaned_names = make_unique_names(
        cleaned_names
    )

    data.columns = cleaned_names

    return data


def make_unique_names(
    names
):
    """
    Make column names unique.
    """

    result = []
    counts = {}

    for name in names:

        if name not in counts:

            counts[name] = 0
            result.append(name)

        else:

            counts[name] += 1

            new_name = (
                f"{name}_{counts[name]}"
            )

            while new_name in counts:

                counts[name] += 1

                new_name = (
                    f"{name}_{counts[name]}"
                )

            counts[new_name] = 0

            result.append(
                new_name
            )

    return result


def detect_datetime_columns(
    df,
    threshold=0.8
):
    """
    Detect object/string columns that
    are likely to contain dates.

    threshold:
        Minimum percentage of values that
        must successfully parse as dates.
    """

    detected = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            detected.append(column)
            continue

        if not (
            pd.api.types.is_object_dtype(
                df[column]
            )
            or
            pd.api.types.is_string_dtype(
                df[column]
            )
        ):
            continue

        series = (
            df[column]
            .dropna()
            .astype(str)
        )

        if series.empty:
            continue

        try:

            parsed = pd.to_datetime(
                series,
                errors="coerce"
            )

            valid_ratio = (
                parsed.notna().mean()
            )

            if valid_ratio >= threshold:

                detected.append(
                    column
                )

        except Exception:

            continue

    return detected


def convert_datetime_columns(
    df,
    columns
):
    """
    Convert selected columns to datetime.
    """

    data = df.copy()

    for column in columns:

        if column not in data.columns:
            continue

        data[column] = pd.to_datetime(
            data[column],
            errors="coerce"
        )

    return data


def get_memory_usage_mb(df):
    """
    Calculate DataFrame memory usage in MB.
    """

    bytes_used = (
        df.memory_usage(
            deep=True
        ).sum()
    )

    return round(
        bytes_used
        / (
            1024 * 1024
        ),
        2
    )


def get_column_type_summary(df):
    """
    Return a DataFrame describing
    each column's data type.
    """

    rows = []

    for column in df.columns:

        series = df[column]

        rows.append({
            "Column": column,
            "Data Type": str(
                series.dtype
            ),
            "Non-Null": int(
                series.notna().sum()
            ),
            "Missing": int(
                series.isna().sum()
            ),
            "Missing %": round(
                series.isna().mean()
                * 100,
                2
            ),
            "Unique": int(
                series.nunique(
                    dropna=True
                )
            )
        })

    return pd.DataFrame(
        rows
    )


def get_numeric_summary(df):
    """
    Return a compact numerical
    summary for the dataset.
    """

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    if len(numeric_columns) == 0:

        return pd.DataFrame()

    summary = (
        df[numeric_columns]
        .describe()
        .T
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "index": "Column"
        }
    )

    return summary


def get_categorical_summary(df):
    """
    Return a compact categorical
    summary for the dataset.
    """

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        ).columns
    )

    if len(categorical_columns) == 0:

        return pd.DataFrame()

    rows = []

    for column in categorical_columns:

        series = df[column]

        mode = series.mode(
            dropna=True
        )

        rows.append({
            "Column": column,
            "Unique Values": int(
                series.nunique(
                    dropna=True
                )
            ),
            "Missing": int(
                series.isna().sum()
            ),
            "Most Common": (
                mode.iloc[0]
                if not mode.empty
                else None
            )
        })

    return pd.DataFrame(
        rows
    )


def format_number(
    value,
    decimals=2
):
    """
    Format numerical values for display.
    """

    if value is None:
        return "N/A"

    try:

        if pd.isna(value):
            return "N/A"

    except Exception:

        pass

    if isinstance(
        value,
        (int, np.integer)
    ):

        return f"{value:,}"

    if isinstance(
        value,
        (float, np.floating)
    ):

        return (
            f"{value:,.{decimals}f}"
        )

    return str(value)


def format_percentage(
    value,
    decimals=2
):
    """
    Format a value as a percentage.
    """

    if value is None:
        return "N/A"

    try:

        if pd.isna(value):
            return "N/A"

    except Exception:

        pass

    return (
        f"{float(value):.{decimals}f}%"
    )


def get_quality_label(
    score
):
    """
    Convert a numerical quality score
    into a readable label.
    """

    if score is None:
        return "Unknown"

    if score >= 90:
        return "Excellent"

    if score >= 75:
        return "Good"

    if score >= 60:
        return "Fair"

    if score >= 40:
        return "Poor"

    return "Critical"


def get_quality_color(
    score
):
    """
    Return a display category based on
    quality score.

    This does not specify UI colors.
    The final app can decide its theme.
    """

    if score is None:
        return "unknown"

    if score >= 90:
        return "excellent"

    if score >= 75:
        return "good"

    if score >= 60:
        return "fair"

    if score >= 40:
        return "poor"

    return "critical"


def generate_download_filename(
    original_filename,
    suffix="cleaned"
):
    """
    Generate a safe filename for downloads.
    """

    if not original_filename:

        original_filename = "dataset.csv"

    name = (
        str(original_filename)
        .rsplit(
            ".",
            1
        )[0]
    )

    name = re.sub(
        r"[^\w\s-]",
        "",
        name
    )

    name = re.sub(
        r"[\s-]+",
        "_",
        name
    )

    name = name.strip("_")

    if not name:

        name = "dataset"

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        f"{name}_{suffix}_{timestamp}.csv"
    )


def dataframe_to_bytes(
    df,
    file_format="csv"
):
    """
    Convert a DataFrame to downloadable bytes.
    """

    if file_format.lower() == "csv":

        return df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

    if file_format.lower() == "excel":

        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            df.to_excel(
                writer,
                index=False,
                sheet_name="Cleaned Data"
            )

        output.seek(0)

        return output.getvalue()

    raise ValueError(
        "Unsupported file format. "
        "Use 'csv' or 'excel'."
    )


def safe_copy(df):
    """
    Create a safe independent copy
    of a DataFrame.
    """

    return df.copy(
        deep=True
    )


def get_dataset_shape_text(df):
    """
    Return a human-readable dataset size.
    """

    rows, columns = df.shape

    return (
        f"{rows:,} rows × "
        f"{columns:,} columns"
    )


def remove_empty_columns(
    df
):
    """
    Remove columns that contain
    no useful values.
    """

    data = df.copy()

    empty_columns = []

    for column in data.columns:

        if data[column].isna().all():

            empty_columns.append(
                column
            )

    if empty_columns:

        data = data.drop(
            columns=empty_columns
        )

    return data, empty_columns


def remove_empty_rows(
    df
):
    """
    Remove completely empty rows.
    """

    data = df.copy()

    before = len(data)

    data = data.dropna(
        how="all"
    )

    removed = (
        before - len(data)
    )

    return data, removed


def get_dataframe_difference(
    original_df,
    new_df
):
    """
    Calculate structural differences
    between two DataFrames.
    """

    original_columns = set(
        original_df.columns
    )

    new_columns = set(
        new_df.columns
    )

    added_columns = sorted(
        new_columns
        - original_columns
    )

    removed_columns = sorted(
        original_columns
        - new_columns
    )

    return {
        "original_rows": len(
            original_df
        ),
        "new_rows": len(
            new_df
        ),
        "rows_changed": (
            len(new_df)
            - len(original_df)
        ),
        "original_columns": len(
            original_columns
        ),
        "new_columns": len(
            new_columns
        ),
        "added_columns": added_columns,
        "removed_columns": removed_columns
    }