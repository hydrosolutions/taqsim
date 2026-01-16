from dataclasses import FrozenInstanceError, dataclass

import pytest

from taqsim.objective.trace import Trace


@dataclass(frozen=True)
class FakeEvent:
    amount: float
    t: int


@dataclass(frozen=True)
class FakeEventWithDeficit:
    deficit: float
    t: int


class TestTraceCreation:
    def test_from_events_extracts_amount_by_default(self):
        events = [FakeEvent(10.0, 0), FakeEvent(20.0, 1), FakeEvent(30.0, 2)]

        trace = Trace.from_events(events)

        assert trace[0] == 10.0
        assert trace[1] == 20.0
        assert trace[2] == 30.0

    def test_from_events_uses_custom_field(self):
        events = [
            FakeEventWithDeficit(deficit=5.0, t=0),
            FakeEventWithDeficit(deficit=15.0, t=1),
        ]

        trace = Trace.from_events(events, field="deficit")

        assert trace[0] == 5.0
        assert trace[1] == 15.0

    def test_from_events_reduces_duplicates_with_sum_by_default(self):
        events = [FakeEvent(10.0, 0), FakeEvent(5.0, 0), FakeEvent(20.0, 1)]

        trace = Trace.from_events(events)

        assert trace[0] == 15.0  # 10.0 + 5.0
        assert trace[1] == 20.0

    def test_from_events_uses_custom_reduce(self):
        events = [FakeEvent(10.0, 0), FakeEvent(5.0, 0), FakeEvent(20.0, 1)]

        trace = Trace.from_events(events, reduce=max)

        assert trace[0] == 10.0  # max(10.0, 5.0)
        assert trace[1] == 20.0

    def test_from_events_returns_empty_for_empty_list(self):
        trace = Trace.from_events([])

        assert len(trace) == 0

    def test_from_dict_creates_trace(self):
        data = {0: 1.0, 1: 2.0, 2: 3.0}

        trace = Trace.from_dict(data)

        assert trace[0] == 1.0
        assert trace[1] == 2.0
        assert trace[2] == 3.0

    def test_constant_creates_uniform_trace(self):
        trace = Trace.constant(5.0, range(3))

        assert trace[0] == 5.0
        assert trace[1] == 5.0
        assert trace[2] == 5.0
        assert len(trace) == 3

    def test_empty_creates_empty_trace(self):
        trace = Trace.empty()

        assert len(trace) == 0


class TestTraceAccess:
    def test_getitem_returns_value(self):
        trace = Trace.from_dict({0: 10.0, 1: 20.0})

        assert trace[0] == 10.0
        assert trace[1] == 20.0

    def test_getitem_raises_key_error(self):
        trace = Trace.from_dict({0: 10.0})

        with pytest.raises(KeyError):
            _ = trace[99]

    def test_len_returns_count(self):
        trace = Trace.from_dict({0: 1.0, 1: 2.0, 2: 3.0})

        assert len(trace) == 3

    def test_iter_yields_sorted_timesteps(self):
        trace = Trace.from_dict({2: 30.0, 0: 10.0, 1: 20.0})

        timesteps = list(trace)

        assert timesteps == [0, 1, 2]

    def test_timesteps_returns_sorted_list(self):
        trace = Trace.from_dict({2: 30.0, 0: 10.0, 1: 20.0})

        assert trace.timesteps() == [0, 1, 2]

    def test_values_returns_in_timestep_order(self):
        trace = Trace.from_dict({2: 30.0, 0: 10.0, 1: 20.0})

        assert trace.values() == [10.0, 20.0, 30.0]

    def test_items_returns_sorted_tuples(self):
        trace = Trace.from_dict({2: 30.0, 0: 10.0, 1: 20.0})

        assert trace.items() == [(0, 10.0), (1, 20.0), (2, 30.0)]

    def test_to_dict_returns_copy(self):
        original_data = {0: 10.0, 1: 20.0}
        trace = Trace.from_dict(original_data)

        result = trace.to_dict()
        result[0] = 999.0  # Mutate the returned dict

        assert trace[0] == 10.0  # Original trace unchanged


class TestTraceTransformation:
    def test_map_applies_function(self):
        trace = Trace.from_dict({0: 1.0, 1: 2.0, 2: 3.0})

        result = trace.map(lambda x: x * 2)

        assert result[0] == 2.0
        assert result[1] == 4.0
        assert result[2] == 6.0

    def test_map_returns_new_trace(self):
        trace = Trace.from_dict({0: 1.0, 1: 2.0})

        result = trace.map(lambda x: x * 2)

        assert trace[0] == 1.0  # Original unchanged
        assert result[0] == 2.0

    def test_filter_keeps_matching(self):
        trace = Trace.from_dict({0: 10.0, 1: 5.0, 2: 20.0, 3: 3.0})

        result = trace.filter(lambda t, v: v > 7)

        assert len(result) == 2
        assert result[0] == 10.0
        assert result[2] == 20.0

    def test_filter_returns_empty_when_none_match(self):
        trace = Trace.from_dict({0: 1.0, 1: 2.0, 2: 3.0})

        result = trace.filter(lambda t, v: v > 100)

        assert len(result) == 0


class TestTraceArithmetic:
    def test_add_traces_intersection_semantics(self):
        trace1 = Trace.from_dict({0: 1.0, 1: 2.0, 2: 3.0})
        trace2 = Trace.from_dict({1: 10.0, 2: 20.0, 3: 30.0})

        result = trace1 + trace2

        assert len(result) == 2  # Only timesteps 1 and 2 are common
        assert result[1] == 12.0
        assert result[2] == 23.0

    def test_add_scalar(self):
        trace = Trace.from_dict({0: 1.0, 1: 2.0})

        result = trace + 10.0

        assert result[0] == 11.0
        assert result[1] == 12.0

    def test_radd_scalar(self):
        trace = Trace.from_dict({0: 1.0, 1: 2.0})

        result = 10.0 + trace

        assert result[0] == 11.0
        assert result[1] == 12.0

    def test_sub_traces(self):
        trace1 = Trace.from_dict({0: 10.0, 1: 20.0})
        trace2 = Trace.from_dict({0: 3.0, 1: 5.0})

        result = trace1 - trace2

        assert result[0] == 7.0
        assert result[1] == 15.0

    def test_sub_scalar(self):
        trace = Trace.from_dict({0: 10.0, 1: 20.0})

        result = trace - 5.0

        assert result[0] == 5.0
        assert result[1] == 15.0

    def test_rsub_scalar(self):
        trace = Trace.from_dict({0: 3.0, 1: 7.0})

        result = 10.0 - trace

        assert result[0] == 7.0
        assert result[1] == 3.0

    def test_mul_traces(self):
        trace1 = Trace.from_dict({0: 2.0, 1: 3.0})
        trace2 = Trace.from_dict({0: 4.0, 1: 5.0})

        result = trace1 * trace2

        assert result[0] == 8.0
        assert result[1] == 15.0

    def test_mul_scalar(self):
        trace = Trace.from_dict({0: 2.0, 1: 3.0})

        result = trace * 10.0

        assert result[0] == 20.0
        assert result[1] == 30.0

    def test_rmul_scalar(self):
        trace = Trace.from_dict({0: 2.0, 1: 3.0})

        result = 10.0 * trace

        assert result[0] == 20.0
        assert result[1] == 30.0

    def test_div_traces(self):
        trace1 = Trace.from_dict({0: 10.0, 1: 20.0})
        trace2 = Trace.from_dict({0: 2.0, 1: 4.0})

        result = trace1 / trace2

        assert result[0] == 5.0
        assert result[1] == 5.0

    def test_div_scalar(self):
        trace = Trace.from_dict({0: 10.0, 1: 20.0})

        result = trace / 2.0

        assert result[0] == 5.0
        assert result[1] == 10.0

    def test_rdiv_scalar(self):
        trace = Trace.from_dict({0: 2.0, 1: 5.0})

        result = 10.0 / trace

        assert result[0] == 5.0
        assert result[1] == 2.0

    def test_neg(self):
        trace = Trace.from_dict({0: 5.0, 1: -3.0})

        result = -trace

        assert result[0] == -5.0
        assert result[1] == 3.0

    def test_pow(self):
        trace = Trace.from_dict({0: 2.0, 1: 3.0})

        result = trace**2

        assert result[0] == 4.0
        assert result[1] == 9.0

    def test_chained_arithmetic(self):
        trace = Trace.from_dict({0: 2.0, 1: 4.0})

        result = (trace * 2 + 1) ** 2

        assert result[0] == 25.0  # (2*2 + 1)^2 = 25
        assert result[1] == 81.0  # (4*2 + 1)^2 = 81


class TestTraceAggregation:
    def test_sum_returns_total(self):
        trace = Trace.from_dict({0: 10.0, 1: 20.0, 2: 30.0})

        assert trace.sum() == 60.0

    def test_sum_empty_returns_zero(self):
        trace = Trace.empty()

        assert trace.sum() == 0.0

    def test_mean_returns_average(self):
        trace = Trace.from_dict({0: 10.0, 1: 20.0, 2: 30.0})

        assert trace.mean() == 20.0

    def test_mean_empty_raises(self):
        trace = Trace.empty()

        with pytest.raises(ValueError, match="cannot compute mean of empty Trace"):
            trace.mean()

    def test_max_returns_maximum(self):
        trace = Trace.from_dict({0: 10.0, 1: 50.0, 2: 30.0})

        assert trace.max() == 50.0

    def test_max_empty_raises(self):
        trace = Trace.empty()

        with pytest.raises(ValueError, match="cannot compute max of empty Trace"):
            trace.max()

    def test_min_returns_minimum(self):
        trace = Trace.from_dict({0: 10.0, 1: 5.0, 2: 30.0})

        assert trace.min() == 5.0

    def test_min_empty_raises(self):
        trace = Trace.empty()

        with pytest.raises(ValueError, match="cannot compute min of empty Trace"):
            trace.min()


class TestTraceCumsum:
    def test_cumsum_returns_running_total(self):
        trace = Trace.from_dict({0: 10.0, 1: 5.0, 2: 15.0})

        result = trace.cumsum()

        assert result.values() == [10.0, 15.0, 30.0]

    def test_cumsum_preserves_timesteps(self):
        trace = Trace.from_dict({0: 10.0, 1: 5.0, 2: 15.0})

        result = trace.cumsum()

        assert result.timesteps() == [0, 1, 2]

    def test_cumsum_returns_new_trace(self):
        trace = Trace.from_dict({0: 10.0, 1: 5.0})

        result = trace.cumsum()

        assert trace[0] == 10.0
        assert trace[1] == 5.0
        assert result[0] == 10.0
        assert result[1] == 15.0

    def test_cumsum_single_value(self):
        trace = Trace.from_dict({5: 42.0})

        result = trace.cumsum()

        assert result[5] == 42.0
        assert len(result) == 1

    def test_cumsum_with_initial_value(self):
        trace = Trace.from_dict({0: 10.0, 1: -3.0, 2: 5.0})

        result = trace.cumsum(initial=50.0)

        assert result.values() == [60.0, 57.0, 62.0]

    def test_cumsum_empty_returns_empty(self):
        trace = Trace.empty()

        result = trace.cumsum()

        assert len(result) == 0

    def test_cumsum_with_negative_values(self):
        trace = Trace.from_dict({0: 10.0, 1: -15.0, 2: 5.0})

        result = trace.cumsum()

        assert result.values() == [10.0, -5.0, 0.0]

    def test_cumsum_with_zeros(self):
        trace = Trace.from_dict({0: 0.0, 1: 5.0, 2: 0.0})

        result = trace.cumsum()

        assert result.values() == [0.0, 5.0, 5.0]

    def test_cumsum_non_contiguous_timesteps(self):
        trace = Trace.from_dict({0: 1.0, 5: 2.0, 10: 3.0})

        result = trace.cumsum()

        assert result.items() == [(0, 1.0), (5, 3.0), (10, 6.0)]

    def test_cumsum_unordered_dict_uses_sorted_order(self):
        trace = Trace.from_dict({2: 3.0, 0: 1.0, 1: 2.0})

        result = trace.cumsum()

        assert result.items() == [(0, 1.0), (1, 3.0), (2, 6.0)]

    def test_cumsum_chainable_with_map(self):
        trace = Trace.from_dict({0: 10.0, 1: 5.0, 2: 15.0})

        result = trace.cumsum().map(lambda x: x * 2)

        assert result.values() == [20.0, 30.0, 60.0]

    def test_cumsum_chainable_with_arithmetic(self):
        trace = Trace.from_dict({0: 10.0, 1: 5.0, 2: 15.0})

        result = trace.cumsum() + 100

        assert result.values() == [110.0, 115.0, 130.0]

    def test_cumsum_last_value_equals_sum_plus_initial(self):
        trace = Trace.from_dict({0: 10.0, 1: 5.0, 2: 15.0})
        initial = 50.0

        result = trace.cumsum(initial=initial)

        assert result.values()[-1] == trace.sum() + initial


class TestTraceImmutability:
    def test_trace_is_frozen(self):
        trace = Trace.from_dict({0: 10.0})

        with pytest.raises(FrozenInstanceError):
            trace._data = {0: 999.0}  # type: ignore[misc]
