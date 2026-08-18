import pandas as pd
import numpy as np


def get_missing_value_summary(df):
    """
    Return missing-value information for every column.
    """

    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isna().sum().values,
        "Missing %": (
            df.isna().mean().values * 100
        ).round(2)
    })

    summary = summary[
        summary["Missing"] > 0
    ].reset_index(drop=True)

    return summary


def fill_missing_values(
    df,
    column,
    method
):
    """
    Fill missing values in a selected column.

    Supported methods:
    - Mean
    - Median
    - Mode
    - Forward Fill
    - Backward Fill
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    cleaned_df = df.copy()

    if cleaned_df[column].isna().sum() == 0:
        return cleaned_df

    if method == "Mean":

        if not pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):
            raise ValueError(
                "Mean can only be used with numerical columns."
            )

        value = cleaned_df[column].mean()

        cleaned_df[column] = (
            cleaned_df[column].fillna(value)
        )

    elif method == "Median":

        if not pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):
            raise ValueError(
                "Median can only be used with numerical columns."
            )

        value = cleaned_df[column].median()

        cleaned_df[column] = (
            cleaned_df[column].fillna(value)
        )

    elif method == "Mode":

        mode = cleaned_df[column].mode()

        if mode.empty:
            raise ValueError(
                f"Unable to determine mode for '{column}'."
            )

        cleaned_df[column] = (
            cleaned_df[column].fillna(
                mode.iloc[0]
            )
        )

    elif method == "Forward Fill":

        cleaned_df[column] = (
            cleaned_df[column].ffill()
        )

    elif method == "Backward Fill":

        cleaned_df[column] = (
            cleaned_df[column].bfill()
        )

    else:

        raise ValueError(
            f"Unsupported missing-value method: {method}"
        )

    return cleaned_df


def fill_all_missing_values(
    df,
    method
):
    """
    Apply a missing-value strategy to
    all applicable columns.

    For numerical columns:
    Mean / Median

    For categorical columns:
    Mode

    Forward / Backward fill:
    Applied to all columns.
    """

    cleaned_df = df.copy()

    numerical_columns = (
        cleaned_df
        .select_dtypes(include="number")
        .columns
    )

    categorical_columns = (
        cleaned_df
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
    )

    if method == "Mean":

        for column in numerical_columns:

            if cleaned_df[column].isna().any():

                cleaned_df[column] = (
                    cleaned_df[column]
                    .fillna(
                        cleaned_df[column].mean()
                    )
                )

    elif method == "Median":

        for column in numerical_columns:

            if cleaned_df[column].isna().any():

                cleaned_df[column] = (
                    cleaned_df[column]
                    .fillna(
                        cleaned_df[column].median()
                    )
                )

    elif method == "Mode":

        for column in cleaned_df.columns:

            if cleaned_df[column].isna().any():

                mode = cleaned_df[column].mode()

                if not mode.empty:

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .fillna(mode.iloc[0])
                    )

    elif method == "Forward Fill":

        cleaned_df = cleaned_df.ffill()

    elif method == "Backward Fill":

        cleaned_df = cleaned_df.bfill()

    else:

        raise ValueError(
            f"Unsupported method: {method}"
        )

    return cleaned_df


def drop_rows_with_missing_values(
    df,
    columns=None
):
    """
    Drop rows containing missing values.

    If columns are provided, only those columns
    are considered.
    """

    cleaned_df = df.copy()

    if columns is None:

        cleaned_df = cleaned_df.dropna()

    else:

        missing_columns = [
            column
            for column in columns
            if column not in cleaned_df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Columns not found: "
                + ", ".join(missing_columns)
            )

        cleaned_df = cleaned_df.dropna(
            subset=columns
        )

    return cleaned_df.reset_index(
        drop=True
    )


def drop_columns_with_missing_values(
    df,
    threshold
):
    """
    Drop columns whose missing percentage
    is greater than the specified threshold.

    Example:
    threshold = 50
    removes columns with more than 50%
    missing values.
    """

    cleaned_df = df.copy()

    missing_percentage = (
        cleaned_df.isna().mean() * 100
    )

    columns_to_drop = (
        missing_percentage[
            missing_percentage > threshold
        ]
        .index
        .tolist()
    )

    cleaned_df = cleaned_df.drop(
        columns=columns_to_drop
    )

    return cleaned_df


def remove_duplicates(
    df,
    subset=None,
    keep="first"
):
    """
    Remove duplicate rows.

    subset:
        Columns to consider when identifying
        duplicates.

    keep:
        'first', 'last', or False.
    """

    cleaned_df = df.copy()

    cleaned_df = cleaned_df.drop_duplicates(
        subset=subset,
        keep=keep
    )

    return cleaned_df.reset_index(
        drop=True
    )


def get_duplicate_rows(
    df,
    subset=None
):
    """
    Return duplicate rows without modifying
    the original dataset.
    """

    if subset is None:

        mask = df.duplicated(
            keep=False
        )

    else:

        mask = df.duplicated(
            subset=subset,
            keep=False
        )

    return df.loc[mask].copy()


def convert_column_type(
    df,
    column,
    target_type
):
    """
    Convert a column to a selected data type.

    Supported types:
    - Numeric
    - String
    - Category
    - Datetime
    - Boolean
    """

    if column not in df.columns:

        raise ValueError(
            f"Column '{column}' does not exist."
        )

    cleaned_df = df.copy()

    if target_type == "Numeric":

        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce"
        )

    elif target_type == "String":

        cleaned_df[column] = (
            cleaned_df[column].astype("string")
        )

    elif target_type == "Category":

        cleaned_df[column] = (
            cleaned_df[column].astype("category")
        )

    elif target_type == "Datetime":

        cleaned_df[column] = pd.to_datetime(
            cleaned_df[column],
            errors="coerce"
        )

    elif target_type == "Boolean":

        cleaned_df[column] = (
            cleaned_df[column]
            .map({
                True: True,
                False: False,
                "True": True,
                "False": False,
                "true": True,
                "false": False,
                1: True,
                0: False,
                "1": True,
                "0": False
            })
        )

    else:

        raise ValueError(
            f"Unsupported target type: {target_type}"
        )

    return cleaned_df


def drop_columns(
    df,
    columns
):
    """
    Remove selected columns from the dataset.
    """

    cleaned_df = df.copy()

    missing_columns = [
        column
        for column in columns
        if column not in cleaned_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Columns not found: "
            + ", ".join(missing_columns)
        )

    return cleaned_df.drop(
        columns=columns
    )


def remove_empty_rows(
    df
):
    """
    Remove rows that contain no data at all.
    """

    cleaned_df = df.copy()

    return cleaned_df.dropna(
        how="all"
    ).reset_index(
        drop=True
    )


def remove_empty_columns(
    df
):
    """
    Remove columns that contain no data at all.
    """

    cleaned_df = df.copy()

    return cleaned_df.dropna(
        axis=1,
        how="all"
    )


def replace_infinite_values(
    df,
    replacement=np.nan
):
    """
    Replace positive and negative infinity
    with a specified value.

    Default:
    infinity -> NaN
    """

    cleaned_df = df.copy()

    numeric_columns = (
        cleaned_df
        .select_dtypes(include="number")
        .columns
    )

    cleaned_df[numeric_columns] = (
        cleaned_df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            replacement
        )
    )

    return cleaned_df


def get_cleaning_summary(
    original_df,
    cleaned_df
):
    """
    Compare the dataset before and after
    cleaning.
    """

    return {
        "original_rows": len(original_df),
        "current_rows": len(cleaned_df),

        "rows_removed": (
            len(original_df)
            - len(cleaned_df)
        ),

        "original_columns": len(
            original_df.columns
        ),

        "current_columns": len(
            cleaned_df.columns
        ),

        "columns_removed": (
            len(original_df.columns)
            - len(cleaned_df.columns)
        ),

        "original_missing": int(
            original_df.isna().sum().sum()
        ),

        "current_missing": int(
            cleaned_df.isna().sum().sum()
        ),

        "missing_values_fixed": max(
            0,
            int(
                original_df.isna().sum().sum()
                - cleaned_df.isna().sum().sum()
            )
        ),

        "original_duplicates": int(
            original_df.duplicated().sum()
        ),

        "current_duplicates": int(
            cleaned_df.duplicated().sum()
        )
    }