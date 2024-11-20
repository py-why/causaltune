from typing import List, Optional

import copy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder, WOEEncoder
from causaltune.data_utils import CausalityDataset


class CausalityDatasetProcessor(BaseEstimator, TransformerMixin):
    """
    A processor for CausalityDataset, designed to preprocess data for causal inference tasks by encoding, normalizing,
    and handling missing values.
    Attributes:
        encoder_type (str): Type of encoder used for categorical feature encoding ('onehot', 'label', 'target', 'woe').
        outcome (str): The target variable used for encoding.
        encoder: Encoder object used during feature transformations.
    """

    def __init__(self):
        """
        Initializes CausalityDatasetProcessor with default attributes for encoder_type, outcome, and encoder.
        """
        self.encoder_type = None
        self.outcome = None
        self.encoder = None

    def fit(
        self,
        cd: CausalityDataset,
        encoder_type: Optional[str] = "onehot",
        outcome: str = None,
    ):
        """
        Fits the processor by preprocessing the input CausalityDataset.
        Args:
            cd (CausalityDataset): The dataset for causal analysis.
            encoder_type (str, optional): Encoder to use for categorical features. Default is 'onehot'.
            outcome (str, optional): The target variable for encoding (needed for 'target' or 'woe'). Default is None.
        Returns:
            CausalityDatasetProcessor: The fitted processor instance.
        """
        cd = copy.deepcopy(cd)
        self.preprocess_dataset(
            cd, encoder_type=encoder_type, outcome=outcome, fit_phase=True
        )
        return self

    def transform(self, cd: CausalityDataset):
        """
        Transforms the CausalityDataset using the fitted encoder.
        Args:
            cd (CausalityDataset): Dataset to transform.
        Returns:
            CausalityDataset: Transformed dataset.
        Raises:
            ValueError: If processor has not been trained yet.
        """
        if self.encoder:
            cd = self.preprocess_dataset(
                cd,
                encoder_type=self.encoder_type,
                outcome=self.outcome,
                fit_phase=False,
            )
            return cd
        else:
            raise ValueError("CausalityDatasetProcessor has not been trained")

    def featurize(
        self,
        cd: CausalityDataset,
        df: pd.DataFrame,
        features: List[str],
        exclude_cols: List[str],
        drop_first: bool = False,
        encoder_type: str = "onehot",
        outcome: str = None,
        fit_phase: bool = True,
    ) -> pd.DataFrame:
        # fill all the NaNs
        categ_columns = []
        for col, t in zip(df.columns, df.dtypes):
            if pd.api.types.is_float_dtype(t):
                df[col] = df[col].fillna(0.0).astype("float32")
            elif pd.api.types.is_integer_dtype(t):
                df[col] = df[col].fillna(-1)
            else:
                df[col] = df[col].fillna("NA").astype("category")
                categ_columns.append(col)

        float_features = [
            f for f in features if pd.api.types.is_float_dtype(df.dtypes[f])
        ]
        float_df = df[float_features].reset_index(drop=True)

        # cast 0/1 int columns to float single-column dummies
        for col, t in zip(df.columns, df.dtypes):
            if pd.api.types.is_integer_dtype(t):
                if len(df[col].unique()) <= 2:
                    df[col] = df[col].fillna(0.0).astype("float32")

        # for other categories, include first column dummy for easier interpretability
        cat_df = df.drop(columns=exclude_cols + float_features)
        if len(cat_df.columns) and encoder_type:
            if encoder_type == "onehot":
                if fit_phase:
                    encoder = OneHotEncoder(
                        cols=categ_columns, drop_invariant=drop_first
                    )
                    dummy_df = encoder.fit_transform(X=cat_df).reset_index(drop=True)
                else:
                    dummy_df = self.encoder.transform(X=cat_df).reset_index(drop=True)
            elif encoder_type == "label":
                if fit_phase:
                    encoder = OrdinalEncoder(cols=categ_columns)
                    dummy_df = encoder.fit_transform(X=cat_df).reset_index(drop=True)
                else:
                    dummy_df = self.encoder.transform(X=cat_df).reset_index(drop=True)
            elif encoder_type == "target":
                if outcome:
                    y = cd.data[outcome]
                else:
                    y = cd.data[cd.outcomes[0]]
                assert (
                    len(set(y)) < 10
                ), "Using TargetEncoder with continuous target is not allowed"
                if fit_phase:
                    encoder = TargetEncoder(cols=categ_columns)
                    dummy_df = encoder.fit_transform(X=cat_df, y=y).reset_index(
                        drop=True
                    )
                else:
                    dummy_df = self.encoder.transform(X=cat_df, y=y).reset_index(
                        drop=True
                    )
            elif encoder_type == "woe":
                if outcome:
                    y = cd.data[outcome]
                else:
                    y = cd.data[cd.outcomes[0]]
                assert (
                    len(set(y)) <= 2
                ), "WOEEncoder: the target column y must be binary"
                if fit_phase:
                    encoder = WOEEncoder(cols=categ_columns)
                    dummy_df = encoder.fit_transform(X=cat_df, y=y).reset_index(
                        drop=True
                    )
                else:
                    dummy_df = self.encoder.transform(X=cat_df, y=y).reset_index(
                        drop=True
                    )
            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
        else:
            encoder = "no"
            dummy_df = pd.DataFrame()

        out = pd.concat(
            [df[exclude_cols].reset_index(drop=True), float_df, dummy_df], axis=1
        )
        if fit_phase:
            self.encoder = encoder
            self.encoder_type = encoder_type
            self.outcome = outcome

        return out

    def preprocess_dataset(
        self,
        cd: CausalityDataset,
        drop_first: Optional[bool] = False,
        fit_phase: bool = True,
        encoder_type: Optional[str] = "onehot",
        outcome: Optional[str] = None,
    ):
        """Preprocesses input dataset for CausalTune by
        converting treatment and instrument columns to integer, normalizing, filling nans, and one-hot encoding.

        Args:
            drop_first (bool): whether to drop the first dummy variable for each categorical feature (default False)
            encoder_type (str): Type of encoder to use for categorical features (default 'onehot').
                Available options are:
                    - 'onehot': OneHotEncoder
                    - 'label': OrdinalEncoder
                    - 'target': TargetEncoder
                    - 'woe': WOEEncoder

        Returns:
            None. Modifies self.data in-place by replacing it with the preprocessed dataframe.
        """

        cd.data[cd.treatment] = cd.data[cd.treatment].astype(int)
        cd.data[cd.instruments] = cd.data[cd.instruments].astype(int)

        # normalize, fill in nans, one-hot encode all the features
        new_chunks = []
        processed_cols = []
        original_columns = cd.data.columns.tolist()
        cols = (
            cd.__dict__["common_causes"]
            + cd.__dict__["effect_modifiers"]
            + cd.__dict__["propensity_modifiers"]
        )
        if cols:
            processed_cols += cols
            re_df = self.featurize(
                cd,
                cd.data[cols],
                features=cols,
                exclude_cols=[],
                drop_first=drop_first,
                fit_phase=fit_phase,
                encoder_type=encoder_type,
                outcome=outcome,
            )
            new_chunks.append(re_df)

        remainder = cd.data[[c for c in cd.data.columns if c not in processed_cols]]
        cd.data = pd.concat([remainder.reset_index(drop=True)] + new_chunks, axis=1)

        # Columns after one-hot encoding
        new_columns = cd.data.columns.tolist()
        fields = ["common_causes", "effect_modifiers", "propensity_modifiers"]
        # Mapping original columns to new (if one-hot) encoded columns
        column_mapping = {}
        for original_col in original_columns:
            matches = [
                col
                for col in new_columns
                if col.startswith(original_col + "_") or original_col == col
            ]
            column_mapping[original_col] = matches
        for col_group in fields:
            updated_columns = []
            for col in cd.__dict__[col_group]:
                updated_columns.extend(column_mapping[col])
            cd.__dict__[col_group] = updated_columns

        return cd
