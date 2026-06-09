"""Helper functions for StackOverflow Developer Salary Analysis."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Columns retained for salary modeling
FEATURE_COLS = [
    "YearsCodePro",
    "Country",
    "EdLevel",
    "DevType",
    "OrgSize",
    "RemoteWork",
    "Employment",
    "AISelect",
]
TARGET_COL = "ConvertedCompYearly"

# USD salary band kept to remove outliers
SALARY_MIN = 10_000
SALARY_MAX = 500_000


def load_data(filepath: str) -> pd.DataFrame:
    """Load the StackOverflow survey CSV into a DataFrame.

    Args:
        filepath: Path to the results CSV file.

    Returns:
        Raw survey DataFrame.
    """
    return pd.read_csv(filepath, low_memory=False)


def clean_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or extreme salary values.

    Keeps only respondents with a ConvertedCompYearly value between
    SALARY_MIN and SALARY_MAX (USD) so that currency conversion
    artefacts and test entries do not skew the model.

    Args:
        df: Raw survey DataFrame.

    Returns:
        DataFrame with valid salary rows only.
    """
    df = df.dropna(subset=[TARGET_COL]).copy()
    mask = df[TARGET_COL].between(SALARY_MIN, SALARY_MAX)
    return df[mask].reset_index(drop=True)


def clean_years_code_pro(df: pd.DataFrame) -> pd.DataFrame:
    """Convert YearsCodePro to a numeric column.

    The survey uses the string "Less than 1 year" for beginners and
    "More than 50 years" for veterans; these are mapped to 0 and 51
    respectively before coercing to float.

    Args:
        df: DataFrame that contains a YearsCodePro column.

    Returns:
        DataFrame with YearsCodePro as float, rows with unparseable
        values dropped.
    """
    df = df.copy()
    replacements = {
        "Less than 1 year": "0",
        "More than 50 years": "51",
    }
    df["YearsCodePro"] = (
        df["YearsCodePro"]
        .replace(replacements)
        .pipe(pd.to_numeric, errors="coerce")
    )
    return df.dropna(subset=["YearsCodePro"]).reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Label-encode a list of categorical columns in-place.

    Missing values are filled with the string "Unknown" before
    encoding so that every row is retained.

    Args:
        df: DataFrame containing the columns to encode.
        cols: List of column names to encode.

    Returns:
        DataFrame with the specified columns replaced by integer codes.
    """
    df = df.copy()
    le = LabelEncoder()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            df[col] = le.fit_transform(df[col])
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Run the full cleaning and encoding pipeline.

    Applies salary cleaning, YearsCodePro parsing, and label encoding
    for all categorical feature columns, then returns X and y ready
    for modelling.

    Args:
        df: Raw survey DataFrame.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the
        salary Series.
    """
    df = clean_salary(df)
    df = clean_years_code_pro(df)

    cat_cols = [
        c for c in FEATURE_COLS if c != "YearsCodePro" and c in df.columns
    ]
    df = encode_categoricals(df, cat_cols)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].fillna(df[available_features].median(numeric_only=True))
    y = df[TARGET_COL]
    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Train a Gradient Boosting Regressor on the prepared features.

    Args:
        X: Feature DataFrame from prepare_features.
        y: Salary target Series.
        test_size: Fraction of data held out for evaluation.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (model, X_test, y_test, y_pred) where model is the
        fitted GradientBoostingRegressor.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred


def evaluate_model(y_test: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute R² and RMSE for a set of predictions.

    Args:
        y_test: True salary values.
        y_pred: Model-predicted salary values.

    Returns:
        Dictionary with keys 'r2' and 'rmse'.
    """
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    return {"r2": round(r2, 4), "rmse": round(rmse, 2)}


def get_feature_importance(model: GradientBoostingRegressor, feature_names: list) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances from the model.

    Args:
        model: Fitted GradientBoostingRegressor.
        feature_names: List of feature column names in training order.

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted
        descending by importance.
    """
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    )
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def get_salary_by_country(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Compute median salary grouped by country for the top N countries.

    Only countries with at least 30 valid salary responses are
    included to avoid unreliable estimates from small samples.

    Args:
        df: Cleaned survey DataFrame with ConvertedCompYearly and Country.
        top_n: Number of top countries (by respondent count) to return.

    Returns:
        DataFrame with columns ['Country', 'MedianSalary', 'Count']
        sorted by MedianSalary descending.
    """
    df = clean_salary(df)
    grouped = (
        df.groupby("Country")[TARGET_COL]
        .agg(MedianSalary="median", Count="count")
        .reset_index()
    )
    grouped = grouped[grouped["Count"] >= 30]
    top_countries = (
        grouped.nlargest(top_n, "Count")["Country"]
    )
    result = grouped[grouped["Country"].isin(top_countries)]
    return result.sort_values("MedianSalary", ascending=False).reset_index(drop=True)


def get_salary_by_ai_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Compare median salaries between AI tool users and non-users.

    Uses the AISelect column which captures whether a respondent is
    currently using AI coding tools.

    Args:
        df: Cleaned survey DataFrame.

    Returns:
        DataFrame with columns ['AISelect', 'MedianSalary', 'Count'].
    """
    df = clean_salary(df)
    df = df.dropna(subset=["AISelect"])
    grouped = (
        df.groupby("AISelect")[TARGET_COL]
        .agg(MedianSalary="median", Count="count")
        .reset_index()
    )
    return grouped.sort_values("MedianSalary", ascending=False).reset_index(drop=True)
