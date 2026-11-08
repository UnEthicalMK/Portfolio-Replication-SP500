import pandas as pd
import pytest

from src.features import partition_data


def test_partition_data_requires_benchmark():
    frame = pd.DataFrame(
        {"ASSET": [0.01, 0.02]},
        index=pd.date_range("2021-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="benchmark"):
        partition_data(frame)
