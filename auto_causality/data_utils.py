from typing import List, Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def featurize(
    df: pd.DataFrame,
    features: List[str],
    exclude_cols: List[str],
    drop_first: bool = False,
    scale_floats: bool = False,
    prune_min_categories: int = 50,
    prune_thresh: float = 0.99,
) -> pd.DataFrame:
    # fill all the NaNs
    for col, t in zip(df.columns, df.dtypes):
        if pd.api.types.is_float_dtype(t):
            df[col] = df[col].fillna(0.0).astype("float32")
        elif pd.api.types.is_integer_dtype(t):
            df[col] = df[col].fillna(-1)
            df[col] = otherize_tail(df[col], -2, prune_thresh, prune_min_categories)
        else:
            df[col] = df[col].fillna("NA")
            df[col] = otherize_tail(
                df[col], "OTHER", prune_thresh, prune_min_categories
            ).astype("category")

    float_features = [f for f in features if pd.api.types.is_float_dtype(df.dtypes[f])]
    if scale_floats:
        float_df = pd.DataFrame(
            RobustScaler().fit_transform(df[float_features]), columns=float_features
        )
    else:
        float_df = df[float_features].reset_index(drop=True)

    # cast 0/1 int columns to float single-column dummies
    for col, t in zip(df.columns, df.dtypes):
        if pd.api.types.is_integer_dtype(t):
            if len(df[col].unique()) <= 2:
                df[col] = df[col].fillna(0.0).astype("float32")

    # for other categories, include first column dummy for easier interpretability
    cat_df = df.drop(columns=exclude_cols + float_features)
    if len(cat_df.columns):
        dummy_df = pd.get_dummies(cat_df, drop_first=drop_first).reset_index(drop=True)
    else:
        dummy_df = pd.DataFrame()

    out = pd.concat(
        [df[exclude_cols].reset_index(drop=True), float_df, dummy_df], axis=1
    )

    return out


def frequent_values(x: pd.Series, thresh: float = 0.99) -> set:
    # get the most frequent values, making up to the fraction thresh of total
    data = x.to_frame("value")
    data["dummy"] = True
    tmp = (
        data[["dummy", "value"]]
        .groupby("value", as_index=False)
        .count()
        .sort_values("dummy", ascending=False)
    )
    tmp["frac"] = tmp.dummy.cumsum() / tmp.dummy.sum()
    return set(tmp["value"][tmp.frac <= thresh].unique())


def otherize_tail(
    x: pd.Series, new_val: Any, thresh: float = 0.99, min_categories: int = 20
):
    uniques = x.unique()
    if len(uniques) < min_categories:
        return x
    else:
        x = x.copy()
        freq = frequent_values(x, thresh)
        x[~x.isin(freq)] = new_val
        return x


def preprocess_dataset(
    data: pd.DataFrame,
    treatment: str,
    targets: Union[str, List[str]],
    instruments: List[str] = [],
    drop_first: bool = False,
    scale_floats: bool = False,
    prune_min_categories: int = 50,
    prune_thresh: float = 0.99,
) -> tuple:
    """preprocesses dataset for causal inference
    Args:
        data (pd.DataFrame): a dataset for causal inference
        treatment: name of treatment column
        targets: target column name or list of target column names

    Returns:
        tuple: dataset, features_x, features_w
    """

    if isinstance(targets, str):
        targets = [targets]

    for c in [treatment] + targets:
        assert c in data.columns, f"Column {c} not found in dataset"

    # else:
    # prepare the data
    features = [c for c in data.columns if c not in [treatment] + targets]

    data[treatment] = data[treatment].astype(int)

    # this is a trick to bypass some DoWhy/EconML bugs
    data["random"] = np.random.randint(0, 2, size=len(data))

    used_df = featurize(
        data,
        features=features,
        exclude_cols=[treatment] + targets,
        drop_first=drop_first,
        scale_floats=scale_floats,
        prune_min_categories=prune_min_categories,
        prune_thresh=prune_thresh,
    )
    used_features = [
        c for c in used_df.columns
        if c not in [treatment] + targets + instruments]

    # Let's treat all features as effect modifiers
    features_X = [f for f in used_features if f != "random"]
    features_W = [f for f in used_features if f not in features_X]

    return used_df, features_X, features_W
