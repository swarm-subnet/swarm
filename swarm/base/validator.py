# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): Miguelik
# Copyright © 2025 Swarm Subnet 124

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


import copy
import time
import numpy as np
import asyncio
import argparse
import os
import threading
import bittensor as bt
from typing import List, Union
from traceback import print_exception
from swarm.base.neuron import BaseNeuron
from swarm.constants import WANDB_IDLE_RESTART_SEC
from swarm.validator.runtime_telemetry import ValidatorRuntimeTracker, tracker_call
from swarm.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from swarm.utils.config import add_validator_args


WEIGHT_SETTER_POLL_SEC = float(os.getenv("SWARM_WEIGHT_SETTER_POLL_SEC", "60"))
WEIGHT_SETTER_RETRY_SEC = float(os.getenv("SWARM_WEIGHT_SETTER_RETRY_SEC", "300"))


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        self.dendrite = bt.Dendrite(wallet=self.wallet)

        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        self.runtime_tracker = ValidatorRuntimeTracker(process_label="validator")

        self._scores_lock = threading.RLock()
        self._set_weights_lock = threading.Lock()
        self._weight_setter_wakeup = threading.Event()
        self._weight_setter_thread: Union[threading.Thread, None] = None
        self._weights_ready_for_setting = False
        self._last_successful_weight_set_block: int | None = None
        self._last_weight_set_attempt_at = 0.0

        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.Axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def _mark_weights_ready_for_setting(self) -> None:
        """Allow background weight submission after backend weights are applied."""
        self._weights_ready_for_setting = True
        self._weight_setter_wakeup.set()

    def _should_set_weights_due(self, *, allow_initial_step: bool = False) -> bool:
        if self.neuron_type == "MinerNeuron":
            return False
        if self.config.neuron.disable_set_weights:
            return False
        if self.step == 0 and not allow_initial_step:
            return False

        current_block = int(self.block)
        last_chain_update = int(self.metagraph.last_update[self.uid])
        if (current_block - last_chain_update) <= self.config.neuron.epoch_length:
            return False

        last_local_set = self._last_successful_weight_set_block
        if last_local_set is not None:
            if (current_block - int(last_local_set)) <= self.config.neuron.epoch_length:
                return False

        return True

    def should_set_weights(self) -> bool:
        return self._should_set_weights_due(allow_initial_step=False)

    def _maybe_set_weights(
        self,
        *,
        source: str,
        allow_initial_step: bool = False,
    ) -> bool:
        if allow_initial_step and not self._weights_ready_for_setting:
            return False
        if not self._should_set_weights_due(allow_initial_step=allow_initial_step):
            return False

        now = time.time()
        if source == "background":
            since_attempt = now - self._last_weight_set_attempt_at
            if since_attempt < WEIGHT_SETTER_RETRY_SEC:
                return False

        if not self._set_weights_lock.acquire(blocking=False):
            bt.logging.info(f"Skipping {source} weight set; another weight set is active")
            return False

        try:
            if not self._should_set_weights_due(allow_initial_step=allow_initial_step):
                return False
            self._last_weight_set_attempt_at = time.time()
            bt.logging.info(f"⚖️ Weight setter triggered ({source})")
            return bool(self.set_weights())
        finally:
            self._set_weights_lock.release()

    def _weight_setter_loop(self) -> None:
        bt.logging.info(
            f"⚖️ Background weight setter started "
            f"(poll={WEIGHT_SETTER_POLL_SEC:.0f}s, retry={WEIGHT_SETTER_RETRY_SEC:.0f}s)"
        )
        while not self.should_exit:
            self._weight_setter_wakeup.wait(WEIGHT_SETTER_POLL_SEC)
            self._weight_setter_wakeup.clear()
            if self.should_exit:
                break
            try:
                self._maybe_set_weights(
                    source="background",
                    allow_initial_step=True,
                )
            except Exception as e:
                bt.logging.error(f"Background weight setter failed: {e}")
        bt.logging.info("⚖️ Background weight setter stopped")

    def _start_weight_setter_thread(self) -> None:
        thread = self._weight_setter_thread
        if thread is not None and thread.is_alive():
            return
        self._weight_setter_wakeup.clear()
        self._weight_setter_thread = threading.Thread(
            target=self._weight_setter_loop,
            name="swarm_weight_setter",
            daemon=True,
        )
        self._weight_setter_thread.start()

    def _stop_weight_setter_thread(self) -> None:
        self._weight_setter_wakeup.set()
        thread = self._weight_setter_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        tracker_call(self, "mark_worker_thread_alive", True)

        # Check that validator is registered on the network.
        self.sync()
        self._start_weight_setter_thread()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                if hasattr(self, "wandb_helper") and self.wandb_helper:
                    if getattr(self, "_completed_evaluation", False):
                        self.wandb_helper.restart()
                        self._completed_evaluation = False
                        self._last_wandb_restart = time.time()
                    elif (time.time() - getattr(self, "_last_wandb_restart", time.time())) >= WANDB_IDLE_RESTART_SEC:
                        self.wandb_helper.restart()
                        self._last_wandb_restart = time.time()
                        bt.logging.info("W&B idle restart (5h cycle)")

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            tracker_call(self, "mark_worker_thread_alive", False)
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            tracker_call(self, "mark_worker_thread_alive", False)
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))
        finally:
            self._stop_weight_setter_thread()
            tracker_call(self, "mark_worker_thread_alive", False)

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            tracker_call(self, "mark_worker_thread_alive", self.thread.is_alive())
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self._weight_setter_wakeup.set()
            self.thread.join(5)
            self._stop_weight_setter_thread()
            self.is_running = False
            tracker_call(self, "mark_worker_thread_alive", False)
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self._weight_setter_wakeup.set()
            self.thread.join(5)
            self._stop_weight_setter_thread()
            self.is_running = False
            tracker_call(self, "mark_worker_thread_alive", False)
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        tracker_call(self, "mark_weights_attempt")
        try:
            scores_lock = getattr(self, "_scores_lock", None)
            if scores_lock is not None:
                with scores_lock:
                    scores_snapshot = np.array(self.scores, copy=True)
            else:
                scores_snapshot = np.array(self.scores, copy=True)

            # Check if self.scores contains any NaN values and log a warning if it does.
            if np.isnan(scores_snapshot).any():
                bt.logging.warning(
                    "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
                )

            # Calculate the average reward for each uid across non-zero values.
            # Replace any NaN values with 0.
            # Compute the norm of the scores
            norm = np.linalg.norm(scores_snapshot, ord=1, axis=0, keepdims=True)

            # Check if the norm is zero or contains NaN values
            if np.any(norm == 0) or np.isnan(norm).any():
                norm = np.ones_like(norm)  # Avoid division by zero or NaN

            # Compute raw_weights safely
            raw_weights = scores_snapshot / norm

            bt.logging.debug("raw_weights", raw_weights)
            bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
            # Process the raw weights to final_weights via subtensor limitations.
            (
                processed_weight_uids,
                processed_weights,
            ) = process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=raw_weights,
                netuid=self.config.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )
            bt.logging.debug("processed_weights", processed_weights)
            bt.logging.debug("processed_weight_uids", processed_weight_uids)

            # Convert to uint16 weights and uids.
            (
                uint_uids,
                uint_weights,
            ) = convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )
            bt.logging.debug("uint_weights", uint_weights)
            bt.logging.debug("uint_uids", uint_uids)

            # Set the weights on chain via our subtensor connection.
            result, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=self.spec_version,
            )
            nonzero_uids = int(np.count_nonzero(scores_snapshot))
            if result is True:
                try:
                    self._last_successful_weight_set_block = int(self.block)
                except Exception:
                    pass
                tracker_call(
                    self,
                    "mark_weights_result",
                    success=True,
                    nonzero_uids=nonzero_uids,
                )
                bt.logging.info("set_weights on chain successfully!")
                return True
            else:
                tracker_call(
                    self,
                    "mark_weights_result",
                    success=False,
                    error=str(msg),
                    nonzero_uids=nonzero_uids,
                )
                bt.logging.error("set_weights failed", msg)
                return False
        except Exception as exc:
            tracker_call(self, "mark_weights_result", success=False, error=str(exc))
            raise

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        with self._scores_lock:
            # Zero out all hotkeys that have been replaced.
            for uid, hotkey in enumerate(self.hotkeys):
                if hotkey != self.metagraph.hotkeys[uid]:
                    self.scores[uid] = 0  # hotkey has been replaced

            # Check to see if the metagraph has changed size.
            # If so, we need to add new hotkeys and moving averages.
            if len(self.hotkeys) < len(self.metagraph.hotkeys):
                # Update the size of the moving average scores.
                new_moving_average = np.zeros((self.metagraph.n))
                min_len = min(len(self.hotkeys), len(self.scores))
                new_moving_average[:min_len] = self.scores[:min_len]
                self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        with self._scores_lock:
            scattered_rewards: np.ndarray = np.zeros_like(self.scores)
            scattered_rewards[uids_array] = rewards
            alpha: float = self.config.neuron.moving_average_alpha
            self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores
        bt.logging.debug(f"Scattered rewards: {rewards}")
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        with self._scores_lock:
            scores = np.array(self.scores, copy=True)
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=scores,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state_path = self.config.neuron.full_path + "/state.npz"
        try:
            state = np.load(state_path)
        except FileNotFoundError:
            bt.logging.warning(f"No state file found at {state_path}, starting fresh.")
            return
        self.step = state["step"]
        with self._scores_lock:
            self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
