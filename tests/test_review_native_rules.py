"""Independent native water rule chemistry checks."""

from fractions import Fraction

from taqsim import (
    CanalLosses,
    CanalSeepageCoefficient,
    Length,
    ReservoirEvaporation,
    SurfaceArea,
    WaterDepth,
    WaterVolume,
)
from taqsim.constituents import ConservativeTransport
from tests.test_conservative_lifecycles import SALT, composition, source, system


def test_native_reservoir_evaporation_exports_all_mobile_salt():
    model = system()
    source(model, "supply", [10])
    rule = ReservoirEvaporation(
        (WaterDepth(2, "m"),) * 12,
        ((WaterVolume(0, "m3"), SurfaceArea(1, "m2")), (WaterVolume(10, "m3"), SurfaceArea(1, "m2"))),
    )
    model.reach("reservoir", "supply", "outlet", rule=rule)
    model.configure_transport(ConservativeTransport((SALT,), boundaries={"supply": (composition(5),)}))
    result = model.build().run(bytes(16)).physical
    assert result.amount("evaporation", 0, view="incoming") == 2
    assert result.amount("evaporation", 0, "salt", view="incoming") == 0
    assert result.amount("outlet", 0, view="incoming") == 8
    assert result.amount("outlet", 0, "salt", view="incoming") == 5
    assert result.concentration("outlet", 0, "salt", view="incoming") == Fraction(5, 8)


def test_native_canal_loss_order_preserves_post_evaporation_mass():
    model = system(frequency="1s")
    source(model, "supply", [16])
    rule = CanalLosses(
        CanalSeepageCoefficient(1, "sqrt(m3/s)/km"),
        Length(1, "km"),
        evaporation_depth=WaterDepth(0.004, "m"),
        width=Length(1, "m"),
        operational_fraction=0.5,
    )
    model.reach("canal", "supply", "outlet", rule=rule)
    model.configure_transport(ConservativeTransport((SALT,), boundaries={"supply": (composition(8),)}))
    result = model.build().run(bytes(16)).physical
    for location, water, mass in [
        ("seepage", 4, 2),
        ("evaporation", 4, 0),
        ("operational-loss", 4, 3),
        ("outlet", 4, 3),
    ]:
        assert result.amount(location, 0, view="incoming") == water
        assert result.amount(location, 0, "salt", view="incoming") == mass
    assert all(balance.residual == 0 for balance in result.basin_balances)
