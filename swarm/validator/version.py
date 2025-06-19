import bittensor as bt
from typing import List
from swarm.utils.weights_version import generate_random_version
from swarm.protocol import TaskSynapse
from swarm.utils.dendrite import dendrite_with_retries
from swarm.utils.logging import ColoredLogger


async def check_miner_not_responding_to_invalid_version(self, task_synapse:TaskSynapse, miner_axons, probability:float, timeout:int):
    try:
        # do_check_versions = random.random() < probability
        do_check_versions = False
        version_responses = []
        if do_check_versions:
            random_version = generate_random_version(self.version, self.least_acceptable_version)
            task_synapse.version = random_version
            ColoredLogger.info(f"Sending check version synapses with random version {random_version}", "yellow")
            responses: List[TaskSynapse] = await dendrite_with_retries(
                dendrite=self.dendrite,
                axons=miner_axons,
                synapse=task_synapse,
                deserialize=True,
                timeout=timeout,
            )
            version_responses.extend(responses)
        else:
            version_responses.extend([TaskSynapse(prompt="", url="", actions=[]) for _ in range(len(miner_axons))])
        return version_responses
    except Exception as e:
        bt.logging.error(f"Error while sending version synapses: {e}")
