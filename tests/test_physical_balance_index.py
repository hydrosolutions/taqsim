"""Physical failure propagation preserves precedence while scaling with records."""

from dataclasses import replace
from decimal import Decimal

from taqsim.constituents import Constituent
from taqsim.physical_results import Balance, BalanceState, PhysicalSample, QualityState, TransferRecord, TransportResult
from taqsim.water_system import TimeAxis


def test_failure_index_preserves_location_day_basin_duplicates_and_precedence():
    samples = tuple(PhysicalSample("reach", i, 10, {"salt": 5}, {"salt": QualityState.SUPPORTED}, {}) for i in range(4))
    unrelated = Balance("elsewhere", 0, "water", 0, 0, 10, 9)
    local = Balance("reach", 1, "salt", 0, 0, 10, 9)
    basin = Balance("basin", 2, "water", 0, 0, 10, 9)
    last = Balance("reach", 2, "salt", 0, 0, 10, 9)
    incomplete = Balance("reach", 3, "water", None, 0, 10, 9)
    balances = (unrelated, local, basin, incomplete)
    basin_balances = (basin,)
    processes = (last,)
    value = TransportResult(
        TimeAxis("2020-01-01", periods=4),
        Decimal(1),
        (Constituent("salt", "NaCl", "NaCl", Decimal(1)),),
        {"reach": samples},
        {"reach": samples},
        {"reach": samples},
        tuple(TransferRecord("reach", "outlet", "main", s.step, "test", s) for s in samples),
        balances,
        basin_balances,
        {},
        processes,
    )
    # Reference is the pre-repair ordered propagation contract, including equal
    # accounts appearing in both local and basin collections.
    expected = []
    for sample in samples:
        quality, reasons = dict(sample.quality), dict(sample.reasons)
        water_quality, water_reason = sample.water_quality, sample.water_reason
        for balance in (*balances, *basin_balances, *processes):
            if balance.step != sample.step or balance.status is not BalanceState.FAILED:
                continue
            if balance not in basin_balances and balance.location != sample.location:
                continue
            if balance.substance == "water":
                water_quality, water_reason = QualityState.UNSUPPORTED, f"failed water balance at {balance.location}"
            for name in quality if balance.substance == "water" else (balance.substance,):
                quality[name] = QualityState.UNSUPPORTED
                reasons[name] = f"failed {balance.substance} balance at {balance.location}"
        expected.append(
            replace(sample, quality=quality, reasons=reasons, water_quality=water_quality, water_reason=water_reason)
        )
    for view in (value.incoming, value.outgoing, value.storage):
        assert view["reach"] == tuple(expected)
    assert tuple(t.sample for t in value.transfers) == tuple(expected)
    assert TransportResult.from_dict(value.to_dict()) == value


def test_balance_inspection_does_not_grow_with_sample_cross_product(monkeypatch):
    days = 200
    calls = 0
    original = Balance.status.fget

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Balance, "status", property(counted))
    samples = tuple(PhysicalSample("reach", i, 10, {}, {}, {}) for i in range(days))
    balances = tuple(Balance("reach", i, "water", 0, 0, 10, 10) for i in range(days))
    TransportResult(
        TimeAxis("2020-01-01", periods=days),
        Decimal(1),
        (),
        {"reach": samples},
        {"reach": samples},
        {"reach": samples},
        (),
        balances,
        (),
        {},
    )
    assert calls <= 2 * days
