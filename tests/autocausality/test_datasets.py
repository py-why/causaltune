import pytest
import subprocess
import sys
import pandas as pd
from auto_causality import datasets


def check_header(df: pd.DataFrame, n_covariates: int):
    """checks if header of dataset is in right format

    Args:
        df (pd.DataFrame): dataset for causal inference, with columns for treatment, outcome and covariates
        n_covariates (int): number of covariates in dataset
    """

    cols = list(df.columns)
    assert "treatment" in cols
    assert "y_factual" in cols
    for i in range(1, n_covariates + 1):
        assert "x" + str(i) in cols
    xcols = [c for c in cols if "x" in c]
    assert len(list(df[xcols].columns)) == n_covariates


def check_preprocessor(df: pd.DataFrame):
    """checks if dataset can be preprocessed (dummy encoding of vars etc...)

    Args:
        df (pd.DataFrame): dataset for causal inference, with cols for treatment, outcome and covariates
    """
    x = datasets.preprocess_dataset(df)
    assert x is not None


class TestDatasets:
    def test_nhefs(self):
        data = datasets.nhefs()
        check_header(data, n_covariates=9)
        check_preprocessor(data)

    def test_lalonde_nsw(self):
        data = datasets.lalonde_nsw()
        check_header(data, n_covariates=8)
        check_preprocessor(data)

    def test_ihdp(self):
        data = datasets.synth_ihdp()
        check_header(data, n_covariates=25)
        check_preprocessor(data)

    def test_acic(self):
        # test defaults:
        data = datasets.synth_acic()
        check_header(data, n_covariates=58)
        check_preprocessor(data)
        # test all conditions:
        for cond in range(1, 11):
            data = datasets.synth_acic(condition=cond)
            check_header(data, n_covariates=58)
            check_preprocessor(data)
        # sanity check
        data = datasets.synth_acic(condition=12)
        assert data is None

    def test_amazon(self):
        # test error handling:
        try:
            import gdown  # noqa F401
        except ImportError:
            data = datasets.amazon_reviews()
            assert data is None
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        finally:
            # test if data can be downloaded and is in right format:
            for rating in ["pos", "neg"]:
                data = datasets.amazon_reviews(rating=rating)
                check_header(data, n_covariates=300)
                check_preprocessor(data)


if __name__ == "__main__":
    pytest.main([__file__])
