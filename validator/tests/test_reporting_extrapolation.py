from swarm.benchmark.engine_parts.reporting import _print_results


def test_print_results_extrapolation_covers_all_challenge_types(capsys):
    _print_results([], [], [], {}, {}, {}, [], elapsed=1.0, eval_start=0.0, num_workers=1)
    assert "type7_office" in capsys.readouterr().out
