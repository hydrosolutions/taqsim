from __future__ import annotations

from tests import make_water_system


def test_the_same_model_and_run_id_have_the_same_authoritative_digest() -> None:
    water_system = make_water_system(3, "1 mL")
    water_system.add_reach("reach", "river", "farm")
    model = water_system.build()
    run_id = bytes(range(16))

    first = model.run(run_id)
    second = model.run(run_id)

    assert first.authoritative_log_digest == second.authoritative_log_digest
    assert first.authoritative_log() == second.authoritative_log()
