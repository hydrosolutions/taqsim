"""Regression tests for NSGA-II Pareto front collapse bug.

The bug occurred when:
1. WaterReleased events were only recorded when actual_release > 0
2. Trace arithmetic used intersection semantics

This caused empty release traces to result in empty cumsum traces,
leading to hydropower=0.0 for many solutions and no dominance,
causing all 100 individuals to end up on the Pareto frontier.
"""

from taqsim.objective.trace import Trace


class TestParetoCollapseRegression:
    """Tests to prevent regression of the Pareto front collapse bug."""

    def test_empty_trace_subtraction_preserves_minuend(self):
        """Subtracting empty trace should preserve all values of the minuend.

        This was the root cause: stored - released where released was empty
        would return an empty trace due to intersection semantics.
        """
        stored = Trace.from_dict({0: 100.0, 1: 150.0, 2: 200.0})
        released = Trace.empty()

        result = stored - released

        assert len(result) == 3
        assert result[0] == 100.0
        assert result[1] == 150.0
        assert result[2] == 200.0

    def test_trace_sub_with_zero_values_preserves_timesteps(self):
        """Subtraction with zero-valued trace should preserve timesteps.

        This tests the fix: WaterReleased events with amount=0 should
        create traces with zero values, not empty traces.
        """
        stored = Trace.from_dict({0: 100.0, 1: 150.0, 2: 200.0})
        released = Trace.from_dict({0: 0.0, 1: 0.0, 2: 0.0})

        result = stored - released

        assert len(result) == 3
        assert result[0] == 100.0
        assert result[1] == 150.0
        assert result[2] == 200.0

    def test_hydropower_calculation_pattern(self):
        """Test the pattern used in hydropower objective calculation.

        The typical pattern is:
        - stored_trace from WaterStored events
        - released_trace from WaterReleased events
        - net_storage = stored - released
        - cumulative = net_storage.cumsum(initial=initial_storage)
        - power = efficiency * cumulative

        If released_trace is empty, this should still work.
        """
        initial_storage = 500.0
        stored = Trace.from_dict({0: 50.0, 1: 30.0, 2: 40.0})
        released = Trace.empty()  # No releases (the bug scenario)

        net_storage = stored - released
        cumulative = net_storage.cumsum(initial=initial_storage)

        # Should have values for all timesteps
        assert len(cumulative) == 3
        # cumsum: initial + stored values
        assert cumulative[0] == 550.0  # 500 + 50
        assert cumulative[1] == 580.0  # 500 + 50 + 30
        assert cumulative[2] == 620.0  # 500 + 50 + 30 + 40

    def test_trace_multiplication_with_empty_trace(self):
        """Multiplying by empty trace should result in zeros, not empty.

        This is relevant for weighted calculations like power = flow * head.
        """
        flow = Trace.from_dict({0: 10.0, 1: 20.0, 2: 30.0})
        head = Trace.empty()

        power = flow * head

        # Should preserve timesteps with zero values
        assert len(power) == 3
        assert power[0] == 0.0
        assert power[1] == 0.0
        assert power[2] == 0.0

    def test_cumsum_after_empty_subtraction_produces_valid_trace(self):
        """Cumsum after subtraction with empty should not be empty.

        This specifically tests the bug path:
        stored_trace - empty_released_trace -> cumsum -> should not be empty
        """
        stored = Trace.from_dict({0: 10.0, 1: 20.0, 2: 30.0})
        released = Trace.empty()

        net = stored - released
        cumulative = net.cumsum()

        assert len(cumulative) == 3
        assert cumulative.sum() > 0  # Not all zeros
