
import pytest

from taqsim.nodes import StorageNode


class TestStorageNode:
    """Test suite for StorageNode class."""

    def setup_method(self):
        """Clear the StorageNode registry before each test."""
        StorageNode.all_ids.clear()

    def test_valid_initialization_with_hv_file(self, simple_hv_csv, simple_evaporation_csv):
        """Test valid initialization with height-volume CSV file."""
        node = StorageNode(
            id="reservoir1",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            evaporation_file=str(simple_evaporation_csv),
            start_year=2020,
            start_month=1,
            num_time_steps=12,
            initial_storage=5000000.0,
            dead_storage=500000.0,
            buffer_coef=0.5,
        )

        assert node.id == "reservoir1"
        assert node.easting == 500000.0
        assert node.northing == 4500000.0
        assert node.capacity == 10000000.0  # Max volume from CSV
        assert node.dead_storage == 500000.0
        assert node.buffer_coef == 0.5
        assert node.storage[0] == 5000000.0
        assert len(node.evaporation_rates) == 12
        assert node.evaporation_rates[0] == 50.0

    def test_initialization_without_evaporation_file(self, simple_hv_csv):
        """Test initialization without evaporation file defaults to zeros."""
        node = StorageNode(
            id="reservoir2",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            num_time_steps=12,
            initial_storage=1000000.0,
        )

        assert len(node.evaporation_rates) == 12
        assert all(rate == 0 for rate in node.evaporation_rates)

    def test_buffer_coef_validation_accepts_zero(self, simple_hv_csv):
        """Test buffer_coef accepts zero."""
        node = StorageNode(
            id="reservoir3", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), buffer_coef=0.0
        )
        assert node.buffer_coef == 0.0

    def test_buffer_coef_validation_accepts_one(self, simple_hv_csv):
        """Test buffer_coef accepts one."""
        node = StorageNode(
            id="reservoir4", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), buffer_coef=1.0
        )
        assert node.buffer_coef == 1.0

    def test_buffer_coef_validation_rejects_negative(self, simple_hv_csv):
        """Test buffer_coef rejects negative values."""
        with pytest.raises(ValueError, match="buffer_coef"):
            StorageNode(
                id="reservoir5", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), buffer_coef=-0.1
            )

    def test_buffer_coef_validation_rejects_greater_than_one(self, simple_hv_csv):
        """Test buffer_coef rejects values greater than one."""
        with pytest.raises(ValueError, match="buffer_coef"):
            StorageNode(
                id="reservoir6", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), buffer_coef=1.5
            )

    def test_dead_storage_validation_accepts_zero(self, simple_hv_csv):
        """Test dead_storage accepts zero."""
        node = StorageNode(
            id="reservoir7", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=0.0
        )
        assert node.dead_storage == 0.0

    def test_dead_storage_validation_rejects_negative(self, simple_hv_csv):
        """Test dead_storage rejects negative values."""
        with pytest.raises(ValueError, match="dead_storage"):
            StorageNode(
                id="reservoir8", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=-1000.0
            )

    def test_initial_storage_validation_accepts_zero(self, simple_hv_csv):
        """Test initial_storage accepts zero."""
        node = StorageNode(
            id="reservoir9", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), initial_storage=0.0
        )
        assert node.storage[0] == 0.0

    def test_initial_storage_validation_rejects_negative(self, simple_hv_csv):
        """Test initial_storage rejects negative values."""
        with pytest.raises(ValueError, match="initial_storage"):
            StorageNode(
                id="reservoir10",
                easting=500000.0,
                northing=4500000.0,
                hv_file=str(simple_hv_csv),
                initial_storage=-1000.0,
            )

    def test_initial_storage_validation_rejects_exceeding_capacity(self, simple_hv_csv):
        """Test initial_storage rejects values exceeding capacity."""
        with pytest.raises(ValueError, match="exceeds maximum capacity"):
            StorageNode(
                id="reservoir11",
                easting=500000.0,
                northing=4500000.0,
                hv_file=str(simple_hv_csv),
                initial_storage=15000000.0,  # Capacity is 10000000.0
            )

    def test_registry_tracking(self, simple_hv_csv):
        """Test StorageNode.all_ids tracks instances."""
        StorageNode.all_ids.clear()

        node1 = StorageNode(id="reservoir12", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv))
        assert "reservoir12" in StorageNode.all_ids

        node2 = StorageNode(id="reservoir13", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv))
        assert "reservoir13" in StorageNode.all_ids
        assert len(StorageNode.all_ids) == 2

    def test_load_hv_data_reads_csv(self, tmp_path):
        """Test _load_hv_data reads CSV with h and v columns."""
        csv_path = tmp_path / "test_hv.csv"
        csv_path.write_text("h,v\n100,0\n110,1000000\n120,3000000\n")

        node = StorageNode(id="reservoir14", easting=500000.0, northing=4500000.0, hv_file=str(csv_path))

        assert node.capacity == 3000000.0
        assert node.hv_data["min_waterlevel"] == 100.0
        assert node.hv_data["max_waterlevel"] == 120.0
        assert node.hv_data["max_depth"] == 20.0

    def test_load_hv_data_sets_capacity_to_max_volume(self, simple_hv_csv):
        """Test capacity is set to maximum volume from CSV."""
        node = StorageNode(id="reservoir15", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv))

        assert node.capacity == 10000000.0

    def test_load_hv_data_missing_h_column(self, tmp_path):
        """Test _load_hv_data raises error for missing h column."""
        csv_path = tmp_path / "bad_hv.csv"
        csv_path.write_text("v\n1000000\n")

        with pytest.raises(ValueError, match="Missing required columns"):
            StorageNode(id="reservoir16", easting=500000.0, northing=4500000.0, hv_file=str(csv_path))

    def test_load_hv_data_missing_v_column(self, tmp_path):
        """Test _load_hv_data raises error for missing v column."""
        csv_path = tmp_path / "bad_hv2.csv"
        csv_path.write_text("h\n100\n")

        with pytest.raises(ValueError, match="Missing required columns"):
            StorageNode(id="reservoir17", easting=500000.0, northing=4500000.0, hv_file=str(csv_path))

    def test_set_release_params_validates_required_params(self, simple_hv_csv):
        """Test set_release_params validates required parameters."""
        node = StorageNode(
            id="reservoir18", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=500000.0
        )

        with pytest.raises(ValueError, match="Missing release parameters"):
            node.set_release_params({"Vr": 100000.0, "V1": 1000000.0})

    def test_set_release_params_converts_single_values_to_lists(self, simple_hv_csv):
        """Test set_release_params converts single values to 12-month lists."""
        node = StorageNode(
            id="reservoir19", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=500000.0
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 5000000.0})

        assert len(node.release_params["Vr"]) == 12
        assert all(v == 100000.0 for v in node.release_params["Vr"])
        assert len(node.release_params["V1"]) == 12
        assert all(v == 1000000.0 for v in node.release_params["V1"])
        assert len(node.release_params["V2"]) == 12
        assert all(v == 5000000.0 for v in node.release_params["V2"])

    def test_set_release_params_accepts_monthly_lists(self, simple_hv_csv):
        """Test set_release_params accepts 12-month lists."""
        node = StorageNode(
            id="reservoir20", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=500000.0
        )

        vr_values = [100000.0 + i * 10000 for i in range(12)]
        v1_values = [1000000.0 + i * 50000 for i in range(12)]
        v2_values = [5000000.0 + i * 100000 for i in range(12)]

        node.set_release_params({"Vr": vr_values, "V1": v1_values, "V2": v2_values})

        assert node.release_params["Vr"] == [float(v) for v in vr_values]
        assert node.release_params["V1"] == [float(v) for v in v1_values]
        assert node.release_params["V2"] == [float(v) for v in v2_values]

    def test_set_release_params_validates_v1_greater_than_dead_storage(self, simple_hv_csv):
        """Test V1 must be greater than dead_storage."""
        node = StorageNode(
            id="reservoir21", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=500000.0
        )

        with pytest.raises(ValueError, match="V1.*must be greater than dead storage"):
            node.set_release_params(
                {
                    "Vr": 100000.0,
                    "V1": 400000.0,  # Less than dead_storage
                    "V2": 5000000.0,
                }
            )

    def test_set_release_params_validates_v1_less_than_v2(self, simple_hv_csv):
        """Test V1 must be less than V2."""
        node = StorageNode(
            id="reservoir22", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=500000.0
        )

        with pytest.raises(ValueError, match="V1.*must be less than V2"):
            node.set_release_params(
                {
                    "Vr": 100000.0,
                    "V1": 6000000.0,
                    "V2": 5000000.0,  # Less than V1
                }
            )

    def test_set_release_params_validates_v2_less_than_or_equal_capacity(self, simple_hv_csv):
        """Test V2 must be less than or equal to capacity."""
        node = StorageNode(
            id="reservoir23", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=500000.0
        )

        with pytest.raises(ValueError, match="V2.*cannot exceed reservoir capacity"):
            node.set_release_params(
                {
                    "Vr": 100000.0,
                    "V1": 1000000.0,
                    "V2": 15000000.0,  # Exceeds capacity of 10000000.0
                }
            )

    def test_set_release_params_validates_vr_non_negative(self, simple_hv_csv):
        """Test Vr cannot be negative."""
        node = StorageNode(
            id="reservoir24", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=500000.0
        )

        with pytest.raises(ValueError, match="Vr.*cannot be negative"):
            node.set_release_params({"Vr": -100000.0, "V1": 1000000.0, "V2": 5000000.0})

    def test_calculate_release_gets_current_month(self, simple_hv_csv):
        """Test calculate_release uses current month from time_step % 12."""
        node = StorageNode(
            id="reservoir25",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            dead_storage=500000.0,
            num_time_steps=24,
        )

        # Use different values for each month to test month selection
        vr_values = [100000.0 + i * 10000.0 for i in range(12)]
        node.set_release_params({"Vr": vr_values, "V1": 1000000.0, "V2": 5000000.0})

        # Test time_step 0 (month 0) - should use Vr[0] = 100000.0
        release_0 = node.calculate_release(3000000.0, 0, 2629800)

        # Test time_step 13 (month 1, wraps around) - should use Vr[1] = 110000.0
        release_13 = node.calculate_release(3000000.0, 13, 2629800)

        # The releases should be different due to different monthly Vr values
        assert release_0 != release_13
        assert release_0 > 0
        assert release_13 > 0

    def test_calculate_release_returns_zero_below_dead_storage(self, simple_hv_csv):
        """Test calculate_release returns 0 when volume <= dead_storage."""
        node = StorageNode(
            id="reservoir26", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv), dead_storage=500000.0
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 5000000.0})

        release = node.calculate_release(400000.0, 0, 2629800)
        assert release == 0

        release = node.calculate_release(500000.0, 0, 2629800)
        assert release == 0

    def test_calculate_release_buffer_zone_reduced_release(self, simple_hv_csv):
        """Test buffer zone applies reduced release based on buffer_coef."""
        node = StorageNode(
            id="reservoir27",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            dead_storage=500000.0,
            buffer_coef=0.5,
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 5000000.0})

        # Volume in buffer zone (between dead_storage and V1)
        release = node.calculate_release(750000.0, 0, 2629800)
        expected = min(0.5 * (750000.0 - 500000.0), 100000.0)
        assert release == expected

    def test_calculate_release_conservation_zone_target_release(self, simple_hv_csv):
        """Test conservation zone returns target release."""
        node = StorageNode(
            id="reservoir28",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            dead_storage=500000.0,
            buffer_coef=0.5,
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 5000000.0})

        # Volume in conservation zone (between V1 and V2)
        release = node.calculate_release(3000000.0, 0, 2629800)
        buffer_contrib = 0.5 * (1000000.0 - 500000.0)
        expected = min(100000.0, 3000000.0 - 1000000.0 + buffer_contrib)
        assert release == expected

    def test_calculate_release_above_conservation_increased_release(self, simple_hv_csv):
        """Test above conservation zone increases release to prevent flooding."""
        node = StorageNode(
            id="reservoir29",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            dead_storage=500000.0,
            buffer_coef=0.5,
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 5000000.0})

        # Add outflow edge for capacity check
        from unittest.mock import Mock

        mock_edge = Mock()
        mock_edge.capacity = 10.0
        node.outflow_edge = mock_edge

        dt = 2629800
        # Volume above V2
        release = node.calculate_release(7000000.0, 0, dt)
        expected = min(max(100000.0, 7000000.0 - 5000000.0), 10.0 * dt)
        assert release == expected

    def test_update_sums_inflows(self, simple_hv_csv):
        """Test update method sums inflows from all edges."""
        from unittest.mock import Mock

        node = StorageNode(
            id="reservoir30",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            initial_storage=5000000.0,
            dead_storage=500000.0,
            num_time_steps=12,
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 8000000.0})

        # Create mock inflow edges
        edge1 = Mock()
        edge1.flow_after_losses = [5.0]
        edge1.source = Mock()
        edge1.source.id = "source1"

        edge2 = Mock()
        edge2.flow_after_losses = [3.0]
        edge2.source = Mock()
        edge2.source.id = "source2"

        node.inflow_edges = {"source1": edge1, "source2": edge2}

        # Create mock outflow edge
        outflow_edge = Mock()
        outflow_edge.capacity = 100.0
        node.outflow_edge = outflow_edge

        dt = 2629800
        node.update(0, dt)

        # Verify storage increased by (5.0 + 3.0) * dt - evaporation - release
        assert len(node.storage) == 2
        assert node.storage[1] > node.storage[0]

    def test_update_calculates_evaporation_loss(self, simple_hv_csv, simple_evaporation_csv):
        """Test update calculates evaporation based on water surface area."""
        from unittest.mock import Mock

        node = StorageNode(
            id="reservoir31",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            evaporation_file=str(simple_evaporation_csv),
            start_year=2020,
            start_month=1,
            num_time_steps=12,
            initial_storage=5000000.0,
            dead_storage=500000.0,
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 8000000.0})

        # Create mock outflow edge
        outflow_edge = Mock()
        outflow_edge.capacity = 100.0
        node.outflow_edge = outflow_edge

        dt = 2629800
        node.update(0, dt)

        # Verify evaporation loss was calculated
        assert len(node.evaporation_losses) == 1
        assert node.evaporation_losses[0] >= 0

    def test_update_handles_spillway_when_exceeding_capacity(self, simple_hv_csv):
        """Test spillway activates when storage exceeds capacity."""
        from unittest.mock import Mock

        node = StorageNode(
            id="reservoir32",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            initial_storage=9500000.0,  # Near capacity
            dead_storage=500000.0,
            num_time_steps=12,
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 8000000.0})

        # Create large inflow to exceed capacity
        edge1 = Mock()
        edge1.flow_after_losses = [100.0]  # Large inflow
        edge1.source = Mock()
        edge1.source.id = "source1"

        node.inflow_edges = {"source1": edge1}

        # Create mock outflow edge
        outflow_edge = Mock()
        outflow_edge.capacity = 10.0
        node.outflow_edge = outflow_edge

        dt = 2629800
        node.update(0, dt)

        # Storage should be capped at capacity
        assert node.storage[1] <= node.capacity

        # Spillway register should record excess
        assert len(node.spillway_register) == 1

    def test_update_updates_outflow_edge(self, simple_hv_csv):
        """Test update method updates the outflow edge with release flow."""
        from unittest.mock import Mock

        node = StorageNode(
            id="reservoir33",
            easting=500000.0,
            northing=4500000.0,
            hv_file=str(simple_hv_csv),
            initial_storage=5000000.0,
            dead_storage=500000.0,
            num_time_steps=12,
        )

        node.set_release_params({"Vr": 100000.0, "V1": 1000000.0, "V2": 8000000.0})

        # Create mock outflow edge
        outflow_edge = Mock()
        outflow_edge.capacity = 100.0
        node.outflow_edge = outflow_edge

        dt = 2629800
        node.update(0, dt)

        # Verify outflow edge was updated
        outflow_edge.update.assert_called_once()
        call_args = outflow_edge.update.call_args[0]
        assert call_args[0] >= 0  # Flow rate should be non-negative

    def test_volume_to_level_interpolation(self, simple_hv_csv):
        """Test volume to level interpolation works correctly."""
        node = StorageNode(id="reservoir34", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv))

        # Test known values
        level = node._volume_to_level(0.0)
        assert level == 100.0

        level = node._volume_to_level(10000000.0)
        assert level == 140.0

        # Test interpolated value
        level = node._volume_to_level(5000000.0)
        assert 120.0 < level < 130.0

    def test_level_to_volume_interpolation(self, simple_hv_csv):
        """Test level to volume interpolation works correctly."""
        node = StorageNode(id="reservoir35", easting=500000.0, northing=4500000.0, hv_file=str(simple_hv_csv))

        # Test known values
        volume = node._level_to_volume(100.0)
        assert volume == 0.0

        volume = node._level_to_volume(140.0)
        assert volume == 10000000.0

        # Test interpolated value
        volume = node._level_to_volume(125.0)
        assert 3000000.0 < volume < 6000000.0
