from __future__ import annotations

from taqsim import Basin


def test_the_same_model_and_run_id_have_the_same_authoritative_digest() -> None:
    basin = Basin(start_date="2020-01-01", timesteps=3)
    basin.add_reach("reach", "river", "farm")
    model = basin.build()
    run_id = bytes(range(16))

    first = model.run(run_id)
    second = model.run(run_id)

    assert first.authoritative_log_digest == second.authoritative_log_digest
    assert first.authoritative_log() == second.authoritative_log()
