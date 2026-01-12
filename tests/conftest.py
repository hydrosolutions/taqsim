from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_csv(tmp_path: Path):
    """Factory fixture for creating temporary CSV files."""

    def _create_csv(data: dict[str, list], filename: str) -> Path:
        df = pd.DataFrame(data)
        csv_path = tmp_path / filename
        df.to_csv(csv_path, index=False)
        return csv_path

    return _create_csv
