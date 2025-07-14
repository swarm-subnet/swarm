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


import sys
import time
import os
from pathlib import Path
from typing import List, Optional

import bittensor as bt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.base.validator import BaseValidatorNeuron
from swarm.validator.forward import forward
from swarm.protocol import ValidationResult, MapTask
import swarm

from loguru import logger

# ─────────── Environment variables loading ─────────────
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# ─────────── Wandb integration ─────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
        
        if WANDB_AVAILABLE:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize wandb run silently."""
        try:
            # Load environment variables for API key
            if DOTENV_AVAILABLE:
                load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))
                bt.logging.debug("Environment variables loaded from .env file")
            
            # Check if API key is available
            api_key = os.getenv('WANDB_API_KEY')
            if not api_key:
                bt.logging.debug("No WANDB_API_KEY found in environment - wandb disabled")
                self.enabled = False
                return
            
            bt.logging.debug(f"WANDB_API_KEY found: {api_key[:10]}...")
            
            # Configure wandb to run silently  
            os.environ["WANDB_SILENT"] = "true"
            os.environ["WANDB_QUIET"] = "true"
            os.environ["WANDB_API_KEY"] = api_key
            
            # Use the specified project and entity
            project_name = "validator-logs"
            entity_name = "swarm-subnet-swarm"
            validator_name = os.getenv('VALIDATOR_NAME', f"validator-{self.validator_uid}")
            run_name = f"{validator_name}-{self.hotkey[:8]}"
            
            wandb_config = {
                "project": project_name,
                "name": run_name,
                "config": {
                    "validator_uid": self.validator_uid,
                    "hotkey": self.hotkey,
                    "version": self.version,
                    "subnet": 124,
                    "neuron_type": "validator"
                },
                "tags": ["validator", "swarm", "subnet-124"],
                "reinit": True,
                "settings": wandb.Settings(
                    quiet=True,
                    show_emoji=False,
                    show_info=False,
                    show_warnings=False
                )
            }
            
            # Add entity (always included)
            wandb_config["entity"] = entity_name
                
            bt.logging.debug(f"Initializing wandb with config: {wandb_config}")
            self.wandb_run = wandb.init(**wandb_config)
            self.enabled = True
            bt.logging.info(f"✅ Wandb run created: {self.wandb_run.name} (project={project_name}, entity={entity_name})")
            
        except Exception as e:
            bt.logging.warning(f"Wandb initialization failed: {e}")
            bt.logging.debug("Validator will continue without wandb logging")
            self.enabled = False

    def log_forward_results(self, 
                          forward_count: int,
                          task: MapTask, 
                          results: List[ValidationResult],
                          timestamp: float) -> None:
        """Log forward pass results silently."""
        if not self.enabled or not self.wandb_run:
            return
            
        try:
            # Calculate aggregate statistics
            if results:
                scores = [r.score for r in results]
                success_count = sum(1 for r in results if r.success)
                
                aggregate_stats = {
                    "forward_count": forward_count,
                    "timestamp": timestamp,
                    "total_miners": len(results),
                    "successful_miners": success_count,
                    "success_rate": success_count / len(results) if results else 0,
                    "max_score": max(scores) if scores else 0,
                    "min_score": min(scores) if scores else 0,
                    "avg_score": sum(scores) / len(scores) if scores else 0,
                    "task_horizon": task.horizon,
                    "task_goal_x": task.goal[0],
                    "task_goal_y": task.goal[1],
                    "task_goal_z": task.goal[2],
                    "obstacles_count": len(getattr(task, 'obstacles', []))
                }
                
                # Add raw validator arrays
                uids = [r.uid for r in results]
                raw_data = {
                    "raw_scores": scores,
                    "raw_uids": uids,
                    "raw_success": [1 if r.success else 0 for r in results],
                    "raw_time_sec": [r.time_sec for r in results],
                    "raw_energy": [r.energy for r in results]
                }
                
                # Combine aggregate stats with raw data
                all_data = {**aggregate_stats, **raw_data}
                self.wandb_run.log(all_data)
                
                # Log individual miner data  
                for result in results:
                    miner_data = {
                        f"miner_{result.uid}/score": result.score,
                        f"miner_{result.uid}/success": 1 if result.success else 0,
                        f"miner_{result.uid}/time_sec": result.time_sec,
                        f"miner_{result.uid}/energy": result.energy,
                        f"miner_{result.uid}/forward_count": forward_count
                    }
                    self.wandb_run.log(miner_data)
            
            else:
                # Log empty forward
                self.wandb_run.log({
                    "forward_count": forward_count,
                    "timestamp": timestamp,
                    "total_miners": 0,
                    "message": "No valid responses from miners"
                })
                
        except Exception:
            # Silent failure
            pass

    def log_weight_update(self, uids: List[int], scores: List[float]) -> None:
        """Log weight updates silently."""
        if not self.enabled or not self.wandb_run:
            return
            
        try:
            weight_data = {
                "weight_update": True,
                "miners_updated": len(uids),
                "total_score": sum(scores) if scores else 0,
                # Raw weight arrays
                "raw_weights": scores,
                "raw_weight_uids": uids
            }
            
            # Individual weight data
            for uid, score in zip(uids, scores):
                weight_data[f"weight_{uid}"] = score
                
            self.wandb_run.log(weight_data)
            
        except Exception:
            # Silent failure
            pass

    def log_error(self, error_message: str, error_type: str = "general") -> None:
        """Log errors silently."""
        if not self.enabled or not self.wandb_run:
            return
            
        try:
            self.wandb_run.log({
                "error": True,
                "error_type": error_type,
                "error_message": error_message
            })
        except Exception:
            # Silent failure
            pass

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

        self.forward_count = 0

        bt.logging.info("load_state()")
        self.load_state()
        
        # Initialize wandb logging
        self.wandb_helper: Optional[WandbHelper] = None
        if WANDB_AVAILABLE:
            try:
                bt.logging.debug("Initializing Wandb helper")
                self.wandb_helper = WandbHelper(
                    validator_uid=self.uid,
                    hotkey=self.wallet.hotkey.ss58_address,
                    version=swarm.__version__
                )
                if self.wandb_helper.enabled:
                    bt.logging.info("✅ Wandb logging enabled")
                else:
                    bt.logging.debug("Wandb helper created but disabled (no API key)")
            except Exception as e:
                bt.logging.debug(f"Wandb initialization failed: {e}")
                self.wandb_helper = None
        else:
            bt.logging.debug("Wandb not available")

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
        """Cleanup wandb helper when validator is destroyed."""
        if hasattr(self, 'wandb_helper') and self.wandb_helper:
            self.wandb_helper.finish()


if __name__ == "__main__":
    # This is Validator Entrypoint
    
    logger.remove()  
    logger.add("logfile.log", level="INFO")  
    logger.add(lambda msg: print(msg, end=""), level="WARNING") 

    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
