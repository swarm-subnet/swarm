# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""The swarm CLI: doctor probes, benchmark and report commands, champion downloads."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from swarm import cli
from swarm.core import sandbox_lockdown


def test_doctor_text_output_with_mocked_checks(monkeypatch, capsys):
    """Doctor exits 0 when every probe passes, and prints one name-and-detail line each."""
    monkeypatch.setattr(
        cli,
        "_run_doctor_checks",
        lambda: [
            cli.DoctorCheck("python", True, "3.11.14", True),
            cli.DoctorCheck("docker_binary", True, "Docker version 26", True),
        ],
    )
    assert cli.main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Swarm Doctor" in output
    assert "python: 3.11.14" in output


def test_doctor_fails_if_required_check_fails(monkeypatch):
    """A failed required probe makes the doctor command exit 1."""
    monkeypatch.setattr(
        cli,
        "_run_doctor_checks",
        lambda: [cli.DoctorCheck("docker_binary", False, "missing", True)],
    )
    assert cli.main(["doctor"]) == 1


def test_doctor_optional_failure_does_not_fail_exit_code(monkeypatch, capsys):
    """An optional probe that fails is still reported, and the exit code stays 0."""
    monkeypatch.setattr(
        cli,
        "_run_doctor_checks",
        lambda: [cli.DoctorCheck("WANDB_API_KEY", False, "not set", False)],
    )
    assert cli.main(["doctor"]) == 0
    assert "WANDB_API_KEY" in capsys.readouterr().out


def test_doctor_checks_runtime_state_dir(monkeypatch):
    """The writable-directory probes cover swarm/state first, then the model directory."""
    captured = []

    def fake_check(path, name):
        """Record the directory and label asked for, and return a passing check for them."""
        captured.append((path, name))
        return cli.DoctorCheck(name, True, str(path), True)

    monkeypatch.setattr(cli, "_check_python_version", lambda: cli.DoctorCheck("python", True, "ok", True))
    monkeypatch.setattr(cli, "_check_docker_binary", lambda: cli.DoctorCheck("docker_binary", True, "ok", True))
    monkeypatch.setattr(cli, "_check_docker_daemon", lambda: cli.DoctorCheck("docker_daemon", True, "ok", True))
    monkeypatch.setattr(cli, "_check_binary_available", lambda name: cli.DoctorCheck(name, True, "ok", True))
    monkeypatch.setattr(cli, "_check_sandbox_lockdown_permissions", lambda: cli.DoctorCheck("sandbox", True, "ok", False))
    monkeypatch.setattr(cli, "_check_module_available", lambda name: cli.DoctorCheck(name, True, "ok", True))
    monkeypatch.setattr(cli, "_check_writable_dir", fake_check)
    monkeypatch.setattr(cli, "_check_submission_template", lambda: cli.DoctorCheck("template", True, "ok", True))
    monkeypatch.setattr(cli, "_check_benchmark_engine", lambda: cli.DoctorCheck("engine", True, "ok", True))
    cli._run_doctor_checks()
    assert captured[0] == (cli.REPO_ROOT / "swarm" / "state", "state_dir")
    assert captured[1][1] == "model_dir"


def test_sandbox_lockdown_permissions_ok_for_root(monkeypatch):
    """Root with nsenter and iptables on PATH passes the network-lockdown probe."""
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    assert cli._check_sandbox_lockdown_permissions().ok is True


def _non_root_with_caps(monkeypatch, caps_by_path: dict) -> None:
    """A non-root process whose iptables is a symlink to xtables-nft-multi, with the given file caps."""
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/sbin/{name}")
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        cli.os.path, "realpath",
        lambda path: "/usr/sbin/xtables-nft-multi" if path == "/usr/sbin/iptables" else path,
    )
    monkeypatch.setattr(sandbox_lockdown, "binary_capabilities", lambda path: caps_by_path.get(path, set()))


def test_sandbox_lockdown_permissions_reads_the_binary_iptables_resolves_to(monkeypatch):
    """Non-root passes only when the nft binary iptables points at holds cap_net_admin, not the legacy one."""
    _non_root_with_caps(monkeypatch, {
        "/usr/sbin/nsenter": {"cap_sys_admin"},
        "/usr/sbin/xtables-legacy-multi": {"cap_net_admin"},
    })
    check = cli._check_sandbox_lockdown_permissions()
    assert check.ok is False
    assert "cap_net_admin to /usr/sbin/xtables-nft-multi" in check.detail

    _non_root_with_caps(monkeypatch, {
        "/usr/sbin/nsenter": {"cap_sys_admin"},
        "/usr/sbin/xtables-nft-multi": {"cap_net_admin", "cap_net_raw"},
    })
    assert cli._check_sandbox_lockdown_permissions().ok is True


def test_benchmark_invokes_engine_directly(monkeypatch, tmp_path):
    """Benchmark calls the engine in-process and forwards --workers into the argv it builds."""
    model_path = tmp_path / "graph.zip"
    model_path.write_bytes(b"zip")
    captured = {}
    monkeypatch.setitem(sys.modules, "swarm.benchmark.engine", SimpleNamespace(main=lambda argv: captured.setdefault("argv", list(argv))))
    assert cli.main(["benchmark", "--model", str(model_path), "--workers", "3"]) == 0
    assert captured["argv"][captured["argv"].index("--workers") + 1] == "3"


def test_benchmark_fails_if_model_missing(capsys, tmp_path):
    """A zip path that does not exist exits 1 and says so on stderr."""
    assert cli.main(["benchmark", "--model", str(tmp_path / "missing.zip")]) == 1
    assert "Model not found" in capsys.readouterr().err


def test_report_text_output_parses_summary(tmp_path, capsys):
    """Report pulls the summary fields out of a benchmark log and prints them as label-value lines."""
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "\n".join([
            "    Seeds evaluated:           50",
            "    Success rate:              40/50 (80.0%)",
            "    Total wall-clock:          120.0s (2.0 min)",
            "    Throughput:                25.00 seeds/min",
            "    Workers used:              3",
            "    Estimated wall-clock:      900.0s (15.0 min)",
        ])
    )
    assert cli.main(["report", "--input", str(log_path)]) == 0
    output = capsys.readouterr().out
    assert "Seeds evaluated: 50" in output
    assert "Workers used: 3" in output


def test_report_text_output_contains_results_block(tmp_path, capsys):
    """A log carrying a RESULTS block has that block echoed back, per-scene rows included."""
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "[17:28:58] === RESULTS ===\n"
        "  type2_open 323521 0.9439 Y\n"
        "    Seeds evaluated:           20\n"
        "    Total wall-clock:          50.0s (0.8 min)\n"
        "    Workers used:              2\n"
        "[17:28:58] === BENCHMARK COMPLETE ===\n"
    )
    assert cli.main(["report", "--input", str(log_path)]) == 0
    output = capsys.readouterr().out
    assert "=== RESULTS ===" in output
    assert "type2_open" in output


def test_extract_benchmark_results_block_strips_progress_noise():
    """The extracted block carries no ANSI escapes and no tqdm progress lines."""
    raw = (
        "\x1b[34mnoise\x1b[0m\n\rSeed progress: 100%|####|\n"
        "[17:28:58] === RESULTS ===\n"
        "    Seeds evaluated:           5\n"
        "[17:28:58] === BENCHMARK COMPLETE ===\n"
    )
    block = cli.extract_benchmark_results_block(raw)
    assert block is not None
    assert "\x1b" not in block
    assert "Seed progress" not in block


def test_report_fails_for_non_report_log(tmp_path):
    """A log with neither a summary nor a RESULTS block exits 1."""
    log_path = tmp_path / "bad.log"
    log_path.write_text("nothing useful here\n")
    assert cli.main(["report", "--input", str(log_path)]) == 1


def test_python_module_entrypoint_help_runs():
    """Invoking the package with -m exits 0 and prints the Swarm CLI banner."""
    result = subprocess.run(
        [sys.executable, "-m", "swarm", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Swarm CLI" in result.stdout


# The doctor check must agree with what the validator actually stages: batch.py
# copies runtime_caps.py into every submission, so a template missing it is broken.


def _template_dir(tmp_path, names):
    """Create swarm/submission_template under tmp_path with one empty file per name, and return it."""
    d = tmp_path / "swarm" / "submission_template"
    d.mkdir(parents=True)
    for name in names:
        (d / name).write_text("")
    return d


def test_doctor_accepts_a_complete_submission_template(tmp_path, monkeypatch):
    """A folder holding all four required files passes the template probe."""
    _template_dir(tmp_path, cli.REQUIRED_TEMPLATE_FILES)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert cli._check_submission_template().ok is True


def test_doctor_rejects_a_template_missing_runtime_caps(tmp_path, monkeypatch):
    """A folder without runtime_caps.py fails the probe, and the detail names the absent file."""
    names = set(cli.REQUIRED_TEMPLATE_FILES) - {"runtime_caps.py"}
    _template_dir(tmp_path, names)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    check = cli._check_submission_template()
    assert check.ok is False
    assert "runtime_caps.py" in check.detail


def test_required_template_files_match_what_the_validator_stages():
    """cli.py and batch.py held two different lists; they must not drift again."""
    staged = (
        Path(__file__).resolve().parents[2]
        / "swarm" / "validator" / "docker" / "docker_evaluator_parts" / "batch.py"
    ).read_text()
    marker = 'for name in ("agent.capnp", "agent_server.py", "main.py", "runtime_caps.py"):'
    assert marker in staged
    assert set(cli.REQUIRED_TEMPLATE_FILES) == {
        "agent.capnp",
        "agent_server.py",
        "main.py",
        "runtime_caps.py",
    }


ZIP_BYTES = b"champion-zip"


def _release(family="cf_search_and_rescue", uid=7, published="2026-10-01T00:00:00Z", score=0.8):
    """Build a public release with the real publisher's tag, body and asset format."""
    digest = hashlib.sha256(ZIP_BYTES).hexdigest()
    return {
        "tag_name": f"{family}-uid{uid}-{digest[:8]}",
        "published_at": published,
        "body": f"- Score: **{score}**\n- Crowned: {published}\n- Archive SHA-256: `{digest}`",
        "assets": [{"name": "submission.zip", "browser_download_url": f"https://github.com/archive/{uid}/submission.zip"}],
    }


def _fake_champion_client(monkeypatch, calls, releases=None, content=ZIP_BYTES, status=200):
    """Serve release pages and archive bytes through a network-free HTTP transport."""
    real_client = httpx.Client

    def respond(request):
        """Record requests and return the requested page or download."""
        calls.append(request)
        if request.url.host == "api.github.com":
            return httpx.Response(status, json=[_release()] if releases is None else releases)
        return httpx.Response(200, content=content)

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))


def test_champion_scopes_release_to_family(monkeypatch, tmp_path):
    """The requested family selects its newest release even when another family was published later."""
    calls = []
    _fake_champion_client(monkeypatch, calls, [
        _release(uid=6, score=0.99),
        _release(uid=7, published="2026-10-02T00:00:00Z"),
        _release("cf_swarm_sar", uid=8, published="2026-10-03T00:00:00Z"),
    ])
    monkeypatch.chdir(tmp_path)
    assert cli.main(["champion", "--family-id", "cf_search_and_rescue"]) == 0
    assert calls[-1].url.path == "/archive/7/submission.zip"
    assert (tmp_path / "champion_cf_search_and_rescue_UID_7.zip").read_bytes() == ZIP_BYTES


def test_champion_without_family_selects_best_current_score(monkeypatch, tmp_path):
    """No family preserves the highest-score default and the original filename."""
    calls = []
    _fake_champion_client(monkeypatch, calls, [
        _release(uid=6, score=1.0),
        _release(uid=7, published="2026-10-02T00:00:00Z", score=0.9),
        _release("cf_swarm_sar", uid=8, published="2026-10-03T00:00:00Z", score=0.8),
    ])
    monkeypatch.chdir(tmp_path)
    assert cli.main(["champion"]) == 0
    assert (tmp_path / "champion_UID_7.zip").exists()


@pytest.mark.parametrize("command", [True, False])
def test_champion_reuses_cached_digest(monkeypatch, tmp_path, command):
    """Both download entry points reuse matching local bytes without fetching the archive."""
    calls = []
    _fake_champion_client(monkeypatch, calls)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "champion_UID_7.zip").write_bytes(ZIP_BYTES)
    if command:
        assert cli.main(["champion"]) == 0
    else:
        assert cli._download_champion_model() == Path("champion_UID_7.zip")
    assert len(calls) == 1


@pytest.mark.parametrize("status,releases,content,message", [
    (200, [], ZIP_BYTES, "No champion release"),
    (403, [], ZIP_BYTES, "rate limit"),
    (429, [], ZIP_BYTES, "rate limit"),
    (200, [_release()], b"wrong zip", "integrity check failed"),
    (200, [{**_release(), "body": "no digest"}], ZIP_BYTES, "valid archive digest"),
    (200, [{**_release(), "assets": []}], ZIP_BYTES, "submission.zip asset"),
    (200, [{**_release(), "tag_name": "cf_search_and_rescue-uid7-00000000"}], ZIP_BYTES, "valid archive digest"),
    (200, [{**_release(), "draft": True}], ZIP_BYTES, "No champion release"),
    (200, [{**_release(), "prerelease": True}], ZIP_BYTES, "No champion release"),
])
def test_champion_download_failures(monkeypatch, tmp_path, capsys, status, releases, content, message):
    """Lookup and integrity failures leave no output and explain the problem."""
    _fake_champion_client(monkeypatch, [], releases, content, status)
    output = tmp_path / "download.zip"
    assert cli.main(["champion", "--output", str(output)]) == 1
    assert not output.exists()
    assert message in capsys.readouterr().err


def test_champion_honors_github_token_and_custom_output(monkeypatch, tmp_path):
    """An optional token authenticates the API request and a custom output is created."""
    calls = []
    _fake_champion_client(monkeypatch, calls)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    output = tmp_path / "nested" / "crown.zip"
    assert cli.main(["champion", "--output", str(output)]) == 0
    assert calls[0].headers["Authorization"] == "Bearer test-token"
    assert output.read_bytes() == ZIP_BYTES


def test_champion_rejects_an_unknown_family(capsys):
    """A family id outside the registered choices dies in argparse before any request goes out."""
    with pytest.raises(SystemExit):
        cli.main(["champion", "--family-id", "cf_nope"])
    assert "invalid choice" in capsys.readouterr().err


def test_benchmark_auto_download_uses_the_benchmark_family(monkeypatch, tmp_path):
    """Benchmark without --model passes its --family-id through to the crown download."""
    model_path = tmp_path / "graph.zip"
    model_path.write_bytes(b"zip")
    captured = {}

    def fake_download(family_id=None):
        """Record the family id it was asked for and return the prepared model path."""
        captured["family_id"] = family_id
        return model_path

    monkeypatch.setattr(cli, "_download_champion_model", fake_download)
    monkeypatch.setitem(sys.modules, "swarm.benchmark.engine", SimpleNamespace(main=lambda argv: 0))

    assert cli.main(["benchmark", "--family-id", "cf_search_and_rescue"]) == 0
    assert captured["family_id"] == "cf_search_and_rescue"


def test_champion_searches_all_release_pages(monkeypatch):
    """A family whose latest release is on a later API page remains discoverable."""
    calls = []
    real_client = httpx.Client

    def respond(request):
        """Return a full unrelated page before the requested family's release."""
        calls.append(request)
        releases = [_release("cf_swarm_sar")] * 100 if request.url.params["page"] == "1" else [_release()]
        return httpx.Response(200, json=releases)

    with real_client(transport=httpx.MockTransport(respond)) as client:
        release = cli._champion_release(client, "cf_search_and_rescue")
    assert release["uid"] == 7
    assert len(calls) == 2


def test_champion_missing_requested_family(monkeypatch, tmp_path, capsys):
    """Releases from another family cannot satisfy an explicit family selection."""
    _fake_champion_client(monkeypatch, [], [_release("cf_swarm_sar")])
    assert cli._download_champion_model("cf_search_and_rescue", tmp_path / "crown.zip") is None
    assert "No champion release exists for cf_search_and_rescue" in capsys.readouterr().err


def test_champion_replaces_invalid_cache(monkeypatch, tmp_path):
    """A stale local archive is replaced only after a valid download."""
    calls = []
    _fake_champion_client(monkeypatch, calls)
    output = tmp_path / "crown.zip"
    output.write_bytes(b"stale")
    assert cli._download_champion_model(output=output) == output
    assert output.read_bytes() == ZIP_BYTES
    assert len(calls) == 2
