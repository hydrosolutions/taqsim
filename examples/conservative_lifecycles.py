"""example : TwoBoundaryCompositions → MixedOutletConcentration."""

from datetime import datetime
from decimal import Decimal
from fractions import Fraction

import polars as pl

from taqsim import ConservationQuantum, IntervalVolume, TimeAxis, WaterSystem
from taqsim.constituents import Composition, ConservativeTransport, Constituent, ConstituentInput, Mass


def main() -> None:
    model = WaterSystem(time=TimeAxis("2020-02-29", 1), quantum=ConservationQuantum.LITRE)
    boundaries = {}
    for name, water, salt in (("tributary-a", 10.0, 8), ("tributary-b", 5.0, 1)):
        frame = pl.DataFrame({"time": [datetime(2020, 2, 29)], "value": [water]})
        model.source(name, IntervalVolume(frame, "m3", "1d", "0.001 m3"))
        model.reach(f"{name}-reach", name, "mix")
        boundaries[name] = (Composition((ConstituentInput("salt", mass=Mass(salt)),)),)
    model.reach("mix", "tributary-a-reach", "outlet")
    model.configure_transport(
        ConservativeTransport(
            (Constituent("salt", "dissolved salt", "as salt", Decimal("0.000001")),),
            boundaries=boundaries,
        )
    )
    result = model.build().run(bytes(16)).physical
    assert result is not None
    assert result.amount("outlet", 0, view="incoming") == 15
    assert result.amount("outlet", 0, "salt", view="incoming") == 9
    assert result.concentration("outlet", 0, "salt", view="incoming") == Fraction(3, 5)
    print("Outlet: 15 m3, 9 kg, 0.6 kg/m3 (600 mg/l)")
    print("Basin residuals:", [(balance.substance, balance.residual) for balance in result.basin_balances])


if __name__ == "__main__":
    main()
