from pathlib import Path

import pandas as pd
import pytest

from taqsim.nodes import DemandNode, HydroWorks, SinkNode, StorageNode
from taqsim.water_system import WaterSystem


@pytest.fixture
def tmp_csv(tmp_path: Path):
    def _create_csv(data: dict[str, list], filename: str) -> Path:
        df = pd.DataFrame(data)
        csv_path = tmp_path / filename
        df.to_csv(csv_path, index=False)
        return csv_path

    return _create_csv


@pytest.fixture
def simple_supply_csv(tmp_csv) -> Path:
    dates = pd.date_range(start="2020-01-01", periods=12, freq="MS")
    data = {"Date": dates, "Q": [10.0] * 12}
    return tmp_csv(data, "supply.csv")


@pytest.fixture
def simple_hv_csv(tmp_csv) -> Path:
    data = {"h": [100.0, 110.0, 120.0, 130.0, 140.0], "v": [0.0, 1000000.0, 3000000.0, 6000000.0, 10000000.0]}
    return tmp_csv(data, "hv.csv")


@pytest.fixture
def simple_demand_csv(tmp_csv) -> Path:
    dates = pd.date_range(start="2020-01-01", periods=12, freq="MS")
    data = {"Date": dates, "test_demand": [5.0] * 12}
    return tmp_csv(data, "demand.csv")


@pytest.fixture
def simple_evaporation_csv(tmp_csv) -> Path:
    dates = pd.date_range(start="2020-01-01", periods=12, freq="MS")
    data = {"Date": dates, "Evaporation": [50.0] * 12}
    return tmp_csv(data, "evaporation.csv")


@pytest.fixture
def simple_rainfall_csv(tmp_csv) -> Path:
    dates = pd.date_range(start="2020-01-01", periods=12, freq="MS")
    data = {"Date": dates, "Precipitation": [100.0] * 12}
    return tmp_csv(data, "rainfall.csv")


@pytest.fixture
def fresh_system() -> WaterSystem:
    StorageNode.all_ids.clear()
    HydroWorks.all_ids.clear()
    DemandNode.all_ids.clear()
    SinkNode.all_ids.clear()
    return WaterSystem(dt=2629800, start_year=2020, start_month=1)
