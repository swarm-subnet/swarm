# The MIT License (MIT)

# Copyright © 2024 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "false")

import subprocess

import bittensor as bt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import wandb
from dotenv import load_dotenv
from loguru import logger

import swarm
from swarm.base.validator import BaseValidatorNeuron
from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator
from swarm.validator.forward import forward


# ───────────────────────────────────────────────────────────────────────────
# Wandb Logging Helper
# ───────────────────────────────────────────────────────────────────────────
class WandbHelper:
    """
    Silent wandb integration for validator logging.
    Logs validator activities without affecting terminal output.
    """

    def __init__(self, validator_uid: int, hotkey: str, version: str = "1.0.0"):
        """Initialize wandb logging silently."""
        self.validator_uid = validator_uid
        self.hotkey = hotkey
        self.version = version
        self.wandb_run = None
        self.enabled = False
        self.cycle_count = 0

        self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize wandb run silently."""
        try:
            # Load environment variables for API key

            load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
            bt.logging.debug("Environment variables loaded from .env file")

            # Check if API key is available
            api_key = os.getenv("WANDB_API_KEY")
            if not api_key:
                bt.logging.debug(
                    "No WANDB_API_KEY found in environment - wandb disabled"
                )
                self.enabled = False
                return

            # Configure wandb to run silently
            os.environ["WANDB_SILENT"] = "true"
            os.environ["WANDB_QUIET"] = "true"
            os.environ["WANDB_API_KEY"] = api_key

            # Use the specified project and entity
            project_name = "validator-logs"
            entity_name = "swarm-subnet-swarm"
            validator_name = os.getenv(
                "VALIDATOR_NAME", f"validator-{self.validator_uid}"
            )
            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            run_name = f"{validator_name}-cycle{self.cycle_count}-{ts}"

            wandb_config = {
                "project": project_name,
                "name": run_name,
                "config": {
                    "validator_uid": self.validator_uid,
                    "hotkey": self.hotkey,
                    "version": self.version,
                    "subnet": 124,
                    "neuron_type": "validator",
                    "cycle_number": self.cycle_count,
                },
                "tags": [
                    "validator",
                    "swarm",
                    "subnet-124",
                    f"cycle-{self.cycle_count}",
                ],
                "reinit": True,
                "settings": wandb.Settings(
                    quiet=True, show_emoji=False, show_info=False, show_warnings=False
                ),
            }

            # Add entity (always included)
            wandb_config["entity"] = entity_name

            bt.logging.debug(f"Initializing wandb with config: {wandb_config}")
            self.wandb_run = wandb.init(**wandb_config)
            self.enabled = True
            bt.logging.info(
                f"✅ Wandb run created: {self.wandb_run.name} (project={project_name}, entity={entity_name})"
            )

        except Exception as e:
            bt.logging.warning(f"Wandb initialization failed: {e}")
            bt.logging.debug("Validator will continue without wandb logging")
            self.enabled = False

    def restart(self) -> None:
        """Restart wandb run after each cycle for cleaner tracking."""
        if not self.enabled:
            return

        try:
            # Finish current run
            if self.wandb_run:
                self.wandb_run.finish(quiet=True)
                bt.logging.debug("Previous wandb run finished")

            # Increment cycle count for next run
            self.cycle_count += 1

            # Start new run
            self._init_wandb()
            if self.enabled and self.wandb_run:
                bt.logging.debug(f"New wandb run started: {self.wandb_run.name}")

        except Exception as e:
            bt.logging.debug(f"Wandb restart failed: {e}")
            self.enabled = False

    def finish(self) -> None:
        """Finish wandb run silently."""
        if self.wandb_run:
            try:
                self.wandb_run.finish(quiet=True)
            except Exception:
                # Silent failure
                pass


# ───────────────────────────────────────────────────────────────────────────
# Main Validator Class
# ───────────────────────────────────────────────────────────────────────────
class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        # Log validator version info
        bt.logging.info(f"🚀 Swarm Validator v{swarm.__version__} started")

        self.forward_count = 0
        self._heartbeat_queue: Optional[list] = None
        self._last_wandb_restart = time.time()

        bt.logging.info("load_state()")
        self.load_state()

        self.wandb_helper: Optional[WandbHelper] = None
        try:
            bt.logging.debug("Initializing Wandb helper")
            self.wandb_helper = WandbHelper(
                validator_uid=self.uid,
                hotkey=self.wallet.hotkey.ss58_address,
                version=swarm.__version__,
            )
            if self.wandb_helper.enabled:
                bt.logging.info("✅ Wandb logging enabled")
            else:
                bt.logging.debug("Wandb helper created but disabled (no API key)")
        except Exception as e:
            bt.logging.debug(f"Wandb initialization failed: {e}")
            self.wandb_helper = None

        # Initialize Docker evaluator at startup
        bt.logging.info("Initializing Docker secure evaluator...")
        try:
            docker_ok = (
                subprocess.run(["docker", "--version"], capture_output=True).returncode
                == 0
            )
            if docker_ok:
                subprocess.run(
                    "docker kill $(docker ps -aq --filter name=swarm_eval_) 2>/dev/null || true",
                    shell=True,
                )
                subprocess.run(
                    "docker kill $(docker ps -aq --filter name=swarm_verify_) 2>/dev/null || true",
                    shell=True,
                )
                subprocess.run(
                    "docker rm -f $(docker ps -aq --filter name=swarm_eval_) 2>/dev/null || true",
                    shell=True,
                )
                subprocess.run(
                    "docker rm -f $(docker ps -aq --filter name=swarm_verify_) 2>/dev/null || true",
                    shell=True,
                )
        except Exception:
            pass
        self.docker_evaluator = DockerSecureEvaluator()
        if DockerSecureEvaluator._base_ready:
            bt.logging.info("✅ Docker evaluator ready")
        else:
            bt.logging.warning("⚠️  Docker evaluator initialization failed")
        try:
            self.docker_evaluator.cleanup()
        except Exception as exc:
            bt.logging.warning(f"Startup evaluation cleanup failed: {exc}")

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)

    def __del__(self):
        """Cleanup wandb helper and docker evaluator when validator is destroyed."""
        if hasattr(self, "docker_evaluator"):
            try:
                self.docker_evaluator.cleanup()
            except Exception:
                pass
        if hasattr(self, "wandb_helper") and self.wandb_helper:
            self.wandb_helper.finish()


if __name__ == "__main__":
    # This is Validator Entrypoint

    logger.remove()
    logger.add("logfile.log", level="INFO")
    logger.add(lambda msg: print(msg, end=""), level="WARNING")

    with Validator() as validator:
        while True:
            if hasattr(validator, 'thread') and not validator.thread.is_alive():
                bt.logging.error("Validator worker thread died! Exiting.")
                break
            time.sleep(5)
