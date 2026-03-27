from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI
from sqlmodel import SQLModel, Session, create_engine, select

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://swarm_user:swarm_password@127.0.0.1:5432/swarm_db",
)
os.environ.setdefault("ADMIN_API_KEY", "test-admin-secret")

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "swarm_backend" / "swarm-backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import api_validators as backend_api_validators  # noqa: E402
from app import auth as backend_auth  # noqa: E402
from app import db as backend_db  # noqa: E402
from app import api_admin as backend_api_admin  # noqa: E402
from app.models import Model, ModelStatus  # noqa: E402
from swarm.validator import backend_api as subnet_backend_api


class _DummyHotkey:
    ss58_address = "validator-hotkey"

    def sign(self, message: bytes) -> bytes:
        _ = message
        return b"\x01\x02"


class _DummyWallet:
    hotkey = _DummyHotkey()


def test_backend_api_client_can_walk_live_backend_contract(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'subnet-backend-integration.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    @contextmanager
    def unlocked_session(*_args, **_kwargs):
        session = Session(engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(backend_db, "engine", engine)
    monkeypatch.setattr(backend_db, "get_session_with_lock", unlocked_session)
    monkeypatch.setattr(backend_api_admin, "get_session_with_lock", unlocked_session)
    monkeypatch.setattr(backend_api_validators, "get_session_with_lock", unlocked_session)
    monkeypatch.setattr(
        backend_api_validators,
        "MODEL_FILES_PATH",
        tmp_path / "models",
    )
    monkeypatch.setattr(backend_api_validators, "get_validator_stake", lambda _hotkey: 5000.0)
    monkeypatch.setattr(backend_api_validators, "get_weights", lambda _session: {41: 1.0})
    monkeypatch.setattr(backend_api_validators, "get_reeval_queue", lambda _session: [])
    monkeypatch.setattr(backend_api_validators, "get_leaderboard_version", lambda _session: 12)
    monkeypatch.setattr(backend_api_validators, "get_benchmark_epoch", lambda: 7)
    monkeypatch.setattr(
        backend_api_validators,
        "get_current_champion",
        lambda session: session.exec(
            select(Model).where(Model.status == ModelStatus.CHAMPION)
        ).first(),
    )

    async def _verified_validator():
        return backend_auth.ValidatorInfo(
            hotkey="validator-hotkey",
            stake=5000.0,
        )

    http_app = FastAPI()
    http_app.include_router(backend_api_validators.router)
    http_app.dependency_overrides = {
        backend_api_validators.get_verified_validator: _verified_validator,
        backend_api_validators.get_verified_validator_upload: _verified_validator,
    }

    now = datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc)
    with Session(engine) as session:
        session.add(
            Model(
                uid=41,
                model_hash="c" * 64,
                coldkey="cold-current",
                hotkey="miner-current",
                status=ModelStatus.CHAMPION,
                benchmark_score=0.98,
                submitted_at=now,
                evaluated_at=now,
            )
        )
        session.commit()

    monkeypatch.setattr(subnet_backend_api, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(
        subnet_backend_api,
        "RUNTIME_STATE_FILE",
        tmp_path / "state" / "runtime_state.json",
    )

    async def _run_flow():
        client = subnet_backend_api.BackendApiClient(
            wallet=_DummyWallet(),
            base_url="http://testserver",
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=http_app),
            base_url=client.base_url,
            timeout=client.timeout,
        )

        model_payload = b"zip-artifact-from-subnet"
        model_hash = hashlib.sha256(model_payload).hexdigest()
        model_path = tmp_path / "model.zip"
        model_path.write_bytes(model_payload)

        try:
            registered = await client.post_new_model(
                uid=42,
                model_hash=model_hash,
                coldkey="cold-new",
                validator_hotkey="validator-hotkey",
                github_url="https://github.com/example/subnet-model",
                miner_hotkey="miner-hotkey",
            )
            assert registered == {"accepted": True, "model_id": 2}

            sync = await client.sync()
            assert sync["current_top"]["uid"] == 41
            assert sync["leaderboard_version"] == 12
            assert sync["benchmark_epoch"] == 7
            assert sync["pending_models"] == [
                {
                    "uid": 42,
                    "model_hash": model_hash,
                    "github_url": "https://github.com/example/subnet-model",
                }
            ]

            uploaded = await client.upload_model_file(42, model_path)
            assert uploaded["stored"] is True
            assert (tmp_path / "models" / f"{model_hash}.zip").read_bytes() == model_payload

            heartbeat = await client.post_heartbeat(
                status="evaluating_benchmark",
                current_uid=42,
                progress=600,
                total_seeds=1000,
            )
            assert heartbeat["recorded"] is True

            published = await client.publish_epoch_seeds(
                epoch_number=7,
                seeds=[11, 12, 13],
                started_at="2026-03-25T10:00:00Z",
                ended_at="2026-03-25T10:10:00Z",
                benchmark_version="v4.0.0",
            )
            assert published["accepted"] is True
        finally:
            await client.close()

    try:
        asyncio.run(_run_flow())
    finally:
        http_app.dependency_overrides = {}
        SQLModel.metadata.drop_all(engine)
