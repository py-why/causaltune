import pytest
from auto_causality.memoizer import MemoizingWrapper
from auto_causality.datasets import synth_ihdp


class TestWrapper:
    def test_wrapper(self):
        m = MemoizingWrapper()
        data_df = synth_ihdp()[:100]
        m.fit(data_df, data_df["y_factual"], time_budget=10, verbose=3)


if __name__ == "__main__":
    pytest.main([__file__])
