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

from __future__ import annotations

from swarm.utils.logging import ColoredLogger, setup_events_logger


def test_setup_events_logger_writes_event_log(tmp_path):
    logger = setup_events_logger(str(tmp_path), events_retention_size=1024 * 1024)
    try:
        logger.event("validator heartbeat")
        for handler in logger.handlers:
            handler.flush()
        content = (tmp_path / "events.log").read_text()
        assert "validator heartbeat" in content
        assert "EVENT" in content
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_colored_logger_wraps_messages(monkeypatch, bt_stub):
    captured = {"info": None, "warning": None, "error": None, "success": None}
    monkeypatch.setattr(bt_stub.logging, "info", lambda msg: captured.__setitem__("info", msg))
    monkeypatch.setattr(bt_stub.logging, "warning", lambda msg: captured.__setitem__("warning", msg))
    monkeypatch.setattr(bt_stub.logging, "error", lambda msg: captured.__setitem__("error", msg))
    monkeypatch.setattr(bt_stub.logging, "success", lambda msg: captured.__setitem__("success", msg))

    ColoredLogger.info("hello", color="green")
    ColoredLogger.warning("warn", color="yellow")
    ColoredLogger.error("boom", color="red")
    ColoredLogger.success("ok", color="blue")

    assert "\033[" in captured["info"]
    assert "\033[" in captured["warning"]
    assert "\033[" in captured["error"]
    assert "\033[" in captured["success"]
    assert captured["info"].endswith("\033[0m")


def test_colored_msg_unknown_color_returns_plain_text():
    assert ColoredLogger._colored_msg("plain", "not-a-color") == "plain"
