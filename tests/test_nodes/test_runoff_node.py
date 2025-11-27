import pytest

from taqsim.nodes import RunoffNode


class TestRunoffNodeConstructor:
    """Test RunoffNode initialization and validation."""

    def test_valid_initialization_with_rainfall_csv(self, simple_rainfall_csv):
        node = RunoffNode(
            id="runoff1",
            area=100.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=simple_rainfall_csv,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        assert node.id == "runoff1"
        assert node.area == 100.0
        assert node.runoff_coefficient == 0.5
        assert node.easting == 500000.0
        assert node.northing == 4000000.0
        assert len(node.rainfall_data) == 12
        assert all(val == 100.0 for val in node.rainfall_data)
        assert node.runoff_history == []
        assert node.outflow_edge is None

    def test_fallback_to_zeros_if_no_csv(self, caplog):
        node = RunoffNode(
            id="runoff2",
            area=50.0,
            runoff_coefficient=0.3,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        assert len(node.rainfall_data) == 12
        assert all(val == 0 for val in node.rainfall_data)
        assert "No rainfall data provided" in caplog.text

    def test_area_validation_must_be_positive(self):
        with pytest.raises(ValueError, match="area must be positive"):
            RunoffNode(
                id="runoff3",
                area=0.0,
                runoff_coefficient=0.5,
                easting=500000.0,
                northing=4000000.0,
                rainfall_csv=None,
                start_year=2020,
                start_month=1,
                num_time_steps=12,
            )

    def test_area_validation_cannot_be_negative(self):
        with pytest.raises(ValueError, match="area must be positive"):
            RunoffNode(
                id="runoff4",
                area=-10.0,
                runoff_coefficient=0.5,
                easting=500000.0,
                northing=4000000.0,
                rainfall_csv=None,
                start_year=2020,
                start_month=1,
                num_time_steps=12,
            )

    def test_runoff_coefficient_validation_in_range(self):
        node = RunoffNode(
            id="runoff5",
            area=100.0,
            runoff_coefficient=0.0,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )
        assert node.runoff_coefficient == 0.0

        node2 = RunoffNode(
            id="runoff6",
            area=100.0,
            runoff_coefficient=1.0,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )
        assert node2.runoff_coefficient == 1.0

    def test_runoff_coefficient_cannot_exceed_one(self):
        with pytest.raises(ValueError, match="runoff_coefficient must be between 0 and 1"):
            RunoffNode(
                id="runoff7",
                area=100.0,
                runoff_coefficient=1.5,
                easting=500000.0,
                northing=4000000.0,
                rainfall_csv=None,
                start_year=2020,
                start_month=1,
                num_time_steps=12,
            )

    def test_runoff_coefficient_cannot_be_negative(self):
        with pytest.raises(ValueError, match="runoff_coefficient must be between 0 and 1"):
            RunoffNode(
                id="runoff8",
                area=100.0,
                runoff_coefficient=-0.1,
                easting=500000.0,
                northing=4000000.0,
                rainfall_csv=None,
                start_year=2020,
                start_month=1,
                num_time_steps=12,
            )

    def test_node_id_validation_empty_string(self):
        with pytest.raises(ValueError, match="ID cannot be empty"):
            RunoffNode(
                id="",
                area=100.0,
                runoff_coefficient=0.5,
                easting=500000.0,
                northing=4000000.0,
                rainfall_csv=None,
                start_year=2020,
                start_month=1,
                num_time_steps=12,
            )

    def test_coordinate_validation_valid_coordinates(self):
        node = RunoffNode(
            id="runoff9",
            area=100.0,
            runoff_coefficient=0.5,
            easting=123456.78,
            northing=9876543.21,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )
        assert node.easting == 123456.78
        assert node.northing == 9876543.21


class TestRunoffNodeAddOutflowEdge:
    """Test RunoffNode outflow edge management."""

    def test_sets_single_outflow_edge(self):
        node = RunoffNode(
            id="runoff10",
            area=100.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        class MockEdge:
            capacity = 50.0

        edge = MockEdge()
        node.add_outflow_edge(edge)

        assert node.outflow_edge is edge

    def test_raises_if_outflow_edge_already_set(self):
        node = RunoffNode(
            id="runoff11",
            area=100.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        class MockEdge:
            capacity = 50.0

        edge1 = MockEdge()
        edge2 = MockEdge()

        node.add_outflow_edge(edge1)

        with pytest.raises(ValueError, match="already has an outflow edge"):
            node.add_outflow_edge(edge2)


class TestRunoffNodeCalculateRunoff:
    """Test RunoffNode runoff calculation logic."""

    def test_converts_rainfall_mm_to_m(self):
        node = RunoffNode(
            id="runoff12",
            area=1.0,
            runoff_coefficient=1.0,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800  # seconds in a month
        rainfall_mm = 1000  # 1000 mm

        runoff = node.calculate_runoff(rainfall_mm, dt)

        # 1000 mm = 1 m
        # 1 m * 1 km² * 1e6 m²/km² * 1.0 coefficient = 1e6 m³
        # 1e6 m³ / 2629800 s = 0.3803 m³/s (approximately)
        expected = (1.0) * 1.0 * 1e6 * 1.0 / dt
        assert abs(runoff - expected) < 1e-6

    def test_multiplies_by_area_km2_to_m2(self):
        node = RunoffNode(
            id="runoff13",
            area=50.0,
            runoff_coefficient=1.0,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800
        rainfall_mm = 100  # 100 mm = 0.1 m

        runoff = node.calculate_runoff(rainfall_mm, dt)

        # 0.1 m * 50 km² * 1e6 m²/km² * 1.0 coefficient = 5e6 m³
        # 5e6 m³ / 2629800 s
        expected = (0.1) * 50.0 * 1e6 * 1.0 / dt
        assert abs(runoff - expected) < 1e-6

    def test_applies_runoff_coefficient(self):
        node = RunoffNode(
            id="runoff14",
            area=100.0,
            runoff_coefficient=0.3,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800
        rainfall_mm = 50  # 50 mm = 0.05 m

        runoff = node.calculate_runoff(rainfall_mm, dt)

        # 0.05 m * 100 km² * 1e6 m²/km² * 0.3 coefficient = 1.5e6 m³
        # 1.5e6 m³ / 2629800 s
        expected = (0.05) * 100.0 * 1e6 * 0.3 / dt
        assert abs(runoff - expected) < 1e-6

    def test_divides_by_dt_to_get_flow_rate(self):
        node = RunoffNode(
            id="runoff15",
            area=10.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        rainfall_mm = 200  # 200 mm = 0.2 m

        # Test with different dt values
        dt1 = 86400  # 1 day in seconds
        runoff1 = node.calculate_runoff(rainfall_mm, dt1)

        dt2 = 2629800  # 1 month in seconds
        runoff2 = node.calculate_runoff(rainfall_mm, dt2)

        # Runoff rate should be inversely proportional to dt
        ratio = runoff1 / runoff2
        expected_ratio = dt2 / dt1
        assert abs(ratio - expected_ratio) < 1e-6

    def test_returns_zero_for_rainfall_zero(self):
        node = RunoffNode(
            id="runoff16",
            area=100.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800
        rainfall_mm = 0

        runoff = node.calculate_runoff(rainfall_mm, dt)

        assert runoff == 0

    def test_returns_zero_for_rainfall_negative(self):
        node = RunoffNode(
            id="runoff17",
            area=100.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800
        rainfall_mm = -10

        runoff = node.calculate_runoff(rainfall_mm, dt)

        assert runoff == 0


class TestRunoffNodeUpdate:
    """Test RunoffNode update method."""

    def test_gets_rainfall_for_time_step(self, simple_rainfall_csv):
        node = RunoffNode(
            id="runoff18",
            area=100.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=simple_rainfall_csv,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        dt = 2629800

        # Rainfall data is [100.0] * 12
        node.update(0, dt)

        assert len(node.runoff_history) == 1

        # Verify runoff is calculated from rainfall[0] = 100.0
        expected = (100.0 / 1000) * 100.0 * 1e6 * 0.5 / dt
        assert abs(node.runoff_history[0] - expected) < 1e-6

    def test_calls_calculate_runoff(self, simple_rainfall_csv):
        node = RunoffNode(
            id="runoff19",
            area=50.0,
            runoff_coefficient=0.4,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=simple_rainfall_csv,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        dt = 2629800
        node.update(5, dt)

        # Verify calculate_runoff was called with correct parameters
        rainfall = node.rainfall_data[5]  # 100.0 mm
        expected = node.calculate_runoff(rainfall, dt)

        assert abs(node.runoff_history[0] - expected) < 1e-6

    def test_stores_in_runoff_history(self, simple_rainfall_csv):
        node = RunoffNode(
            id="runoff20",
            area=100.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=simple_rainfall_csv,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        dt = 2629800

        assert len(node.runoff_history) == 0

        node.update(0, dt)
        assert len(node.runoff_history) == 1

        node.update(1, dt)
        assert len(node.runoff_history) == 2

        node.update(2, dt)
        assert len(node.runoff_history) == 3

    def test_updates_outflow_edge_capped_by_capacity(self, simple_rainfall_csv):
        node = RunoffNode(
            id="runoff21",
            area=1000.0,
            runoff_coefficient=0.8,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=simple_rainfall_csv,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        class MockEdge:
            def __init__(self):
                self.capacity = 100.0
                self.updated_flows = []

            def update(self, flow):
                self.updated_flows.append(flow)

        edge = MockEdge()
        node.add_outflow_edge(edge)

        dt = 2629800
        node.update(0, dt)

        # Calculate expected runoff
        rainfall = 100.0  # mm
        runoff = (rainfall / 1000) * 1000.0 * 1e6 * 0.8 / dt

        # Edge should receive min(runoff, capacity)
        expected_flow = min(runoff, edge.capacity)

        assert len(edge.updated_flows) == 1
        assert abs(edge.updated_flows[0] - expected_flow) < 1e-6

    def test_updates_outflow_edge_below_capacity(self, simple_rainfall_csv):
        node = RunoffNode(
            id="runoff22",
            area=10.0,
            runoff_coefficient=0.2,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=simple_rainfall_csv,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        class MockEdge:
            def __init__(self):
                self.capacity = 10000.0
                self.updated_flows = []

            def update(self, flow):
                self.updated_flows.append(flow)

        edge = MockEdge()
        node.add_outflow_edge(edge)

        dt = 2629800
        node.update(0, dt)

        # Calculate expected runoff
        rainfall = 100.0  # mm
        runoff = (rainfall / 1000) * 10.0 * 1e6 * 0.2 / dt

        # Runoff should be well below capacity
        assert runoff < edge.capacity

        # Edge should receive full runoff
        assert len(edge.updated_flows) == 1
        assert abs(edge.updated_flows[0] - runoff) < 1e-6

    def test_update_without_outflow_edge(self, simple_rainfall_csv):
        node = RunoffNode(
            id="runoff23",
            area=100.0,
            runoff_coefficient=0.5,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=simple_rainfall_csv,
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        dt = 2629800

        # Should not raise error even without outflow edge
        node.update(0, dt)

        assert len(node.runoff_history) == 1
        assert node.runoff_history[0] > 0


class TestRunoffNodeUnitConversions:
    """Test unit conversions with known values."""

    def test_known_value_simple_case(self):
        # Simple case: 1 km², 1000 mm rainfall, coefficient 1.0
        node = RunoffNode(
            id="runoff24",
            area=1.0,
            runoff_coefficient=1.0,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800  # seconds in month
        rainfall_mm = 1000  # 1000 mm = 1 m

        runoff = node.calculate_runoff(rainfall_mm, dt)

        # 1 m * 1 km² = 1 m * 1e6 m² = 1e6 m³
        # 1e6 m³ / 2629800 s ≈ 0.3803 m³/s
        expected = 1e6 / dt
        assert abs(runoff - expected) < 1e-6

    def test_known_value_realistic_case(self):
        # Realistic case: 100 km² catchment, 50 mm monthly rainfall, 0.4 coefficient
        node = RunoffNode(
            id="runoff25",
            area=100.0,
            runoff_coefficient=0.4,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800
        rainfall_mm = 50  # 50 mm = 0.05 m

        runoff = node.calculate_runoff(rainfall_mm, dt)

        # 0.05 m * 100 km² * 1e6 m²/km² * 0.4 = 2e6 m³
        # 2e6 m³ / 2629800 s ≈ 0.7607 m³/s
        volume = 0.05 * 100.0 * 1e6 * 0.4
        expected = volume / dt
        assert abs(runoff - expected) < 1e-6

    def test_zero_coefficient_produces_zero_runoff(self):
        node = RunoffNode(
            id="runoff26",
            area=500.0,
            runoff_coefficient=0.0,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800
        rainfall_mm = 200

        runoff = node.calculate_runoff(rainfall_mm, dt)

        assert runoff == 0

    def test_full_coefficient_no_losses(self):
        node = RunoffNode(
            id="runoff27",
            area=25.0,
            runoff_coefficient=1.0,
            easting=500000.0,
            northing=4000000.0,
            rainfall_csv=None,
            start_year=2020,
            start_month=1,
            num_time_steps=1,
        )

        dt = 2629800
        rainfall_mm = 80  # 80 mm = 0.08 m

        runoff = node.calculate_runoff(rainfall_mm, dt)

        # 0.08 m * 25 km² * 1e6 m²/km² * 1.0 = 2e6 m³
        volume = 0.08 * 25.0 * 1e6 * 1.0
        expected = volume / dt
        assert abs(runoff - expected) < 1e-6
