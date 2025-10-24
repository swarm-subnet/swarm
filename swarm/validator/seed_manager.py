from datetime import datetime, timezone, timedelta
import hashlib
import time
from typing import Tuple
import bittensor as bt


class SynchronizedSeedManager:

    def __init__(self, secret_key: str, window_minutes: int = 10):
        self.secret_key = secret_key
        self.window_minutes = window_minutes
        self._last_window_start = None

    def get_current_window(self) -> Tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        minutes_floored = (now.minute // self.window_minutes) * self.window_minutes
        window_start = now.replace(minute=minutes_floored, second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=self.window_minutes)
        return window_start, window_end

    def generate_seed(self) -> Tuple[int, datetime, datetime]:
        window_start, window_end = self.get_current_window()

        time_string = window_start.strftime("%Y-%m-%d-%H:%M")
        seed_input = f"{self.secret_key}{time_string}"
        hash_object = hashlib.sha256(seed_input.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        seed = int(hash_hex[:8], 16)

        if self._last_window_start != window_start:
            self._last_window_start = window_start
            bt.logging.info(
                f"New seed window: {window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')} UTC | "
                f"Seed: {seed}"
            )

        return seed, window_start, window_end

    def should_wait(self) -> bool:
        _, window_end = self.get_current_window()
        now = datetime.now(timezone.utc)
        return now < window_end

    def wait_for_next_window(self) -> None:
        _, window_end = self.get_current_window()
        now = datetime.now(timezone.utc)

        if now < window_end:
            sleep_seconds = (window_end - now).total_seconds()
            bt.logging.info(
                f"Cycle complete. Waiting {sleep_seconds:.1f}s until "
                f"{window_end.strftime('%H:%M:%S')} UTC"
            )
            time.sleep(sleep_seconds + 0.5)
