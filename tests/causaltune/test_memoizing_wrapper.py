import pytest
from causaltune.memoizer import MemoizingWrapper
from causaltune.datasets import generate_synthetic_data


class TestWrapper:
    def test_wrapper(self):
        m = MemoizingWrapper()
        data_df = generate_synthetic_data().data
        data_df = data_df[:100]
        m.fit(data_df, data_df["y_factual"], time_budget=10, verbose=3)


if __name__ == "__main__":
    pytest.main([__file__])
