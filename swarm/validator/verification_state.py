import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import bittensor as bt

from swarm.constants import (
    VERIFICATION_STATE_FILE,
    VERIFICATION_CACHE_DAYS,
)


class VerificationState:

    def __init__(self, state_file: Path = VERIFICATION_STATE_FILE):
        self.state_file = state_file
        self._ensure_file()

    def _ensure_file(self):
        """Ensure state file exists"""
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_state({"verifications": {}, "queue": []})

    def _load_state(self) -> Dict:
        """Load state from file"""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            bt.logging.warning(f"Failed to load verification state: {e}")
            return {"verifications": {}, "queue": []}

    def _save_state(self, state: Dict):
        """Save state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            bt.logging.error(f"Failed to save verification state: {e}")

    def get_status(self, uid: int) -> Optional[str]:
        """Get verification status for UID"""
        state = self._load_state()
        uid_str = str(uid)
        if uid_str in state["verifications"]:
            verification = state["verifications"][uid_str]
            status = verification.get("status")

            # Only check expiration for verified status
            if status == "verified":
                valid_until_str = verification.get("valid_until", "2000-01-01T00:00:00Z").replace("Z", "")
                valid_until = datetime.fromisoformat(valid_until_str)
                if datetime.utcnow() >= valid_until:
                    return None

            return status
        return None

    def set_verified(self, uid: int, score: float, details: Dict):
        """Mark UID as verified"""
        state = self._load_state()
        uid_str = str(uid)

        now = datetime.utcnow()
        valid_until = now + timedelta(days=VERIFICATION_CACHE_DAYS)

        state["verifications"][uid_str] = {
            "status": "verified",
            "score": score,
            "last_check": now.isoformat() + "Z",
            "valid_until": valid_until.isoformat() + "Z",
            "details": details,
            "failure_count": 0,
        }

        if uid in state["queue"]:
            state["queue"].remove(uid)

        self._save_state(state)
        bt.logging.info(f"UID {uid} marked as verified (score: {score:.2f})")

    def set_failed(self, uid: int, reason: str):
        """Mark UID as failed verification"""
        state = self._load_state()
        uid_str = str(uid)

        if uid_str in state["verifications"]:
            failure_count = state["verifications"][uid_str].get("failure_count", 0) + 1
        else:
            failure_count = 1

        state["verifications"][uid_str] = {
            "status": "failed",
            "score": 0.0,
            "last_check": datetime.utcnow().isoformat() + "Z",
            "valid_until": datetime.utcnow().isoformat() + "Z",
            "failure_count": failure_count,
            "failure_reason": reason,
        }

        if uid in state["queue"]:
            state["queue"].remove(uid)

        self._save_state(state)
        bt.logging.warning(f"UID {uid} verification failed (attempt {failure_count}): {reason}")

    def set_pending(self, uid: int):
        """Mark UID as pending verification"""
        state = self._load_state()
        uid_str = str(uid)

        if uid_str not in state["verifications"]:
            state["verifications"][uid_str] = {
                "status": "pending",
                "score": 0.0,
                "last_check": datetime.utcnow().isoformat() + "Z",
                "valid_until": datetime.utcnow().isoformat() + "Z",
            }

        if uid not in state["queue"]:
            state["queue"].append(uid)

        self._save_state(state)

    def get_queue(self) -> List[int]:
        """Get list of UIDs pending verification"""
        state = self._load_state()
        return state.get("queue", [])

    def remove_from_queue(self, uid: int):
        """Remove UID from verification queue"""
        state = self._load_state()
        if uid in state["queue"]:
            state["queue"].remove(uid)
            self._save_state(state)

    def get_verification_details(self, uid: int) -> Optional[Dict]:
        """Get full verification details for UID"""
        state = self._load_state()
        uid_str = str(uid)
        return state["verifications"].get(uid_str)

    def cleanup_expired(self):
        """Remove expired verifications from state"""
        state = self._load_state()
        now = datetime.utcnow()

        expired_uids = []
        for uid_str, verification in state["verifications"].items():
            valid_until_str = verification.get("valid_until", "2000-01-01T00:00:00Z").replace("Z", "")
            valid_until = datetime.fromisoformat(valid_until_str)
            if now > valid_until and verification.get("status") == "verified":
                expired_uids.append(uid_str)

        for uid_str in expired_uids:
            del state["verifications"][uid_str]

        if expired_uids:
            self._save_state(state)
            bt.logging.debug(f"Cleaned up {len(expired_uids)} expired verifications")
