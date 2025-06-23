from swarm.protocol import TaskSynapse, SetOperatorEndpointSynapse
from swarm.utils.logging import ColoredLogger
import copy
import random
import bittensor as bt
from typing import List

async def dendrite_with_retries(
    dendrite: bt.dendrite,
    axons: list,
    synapse: TaskSynapse | SetOperatorEndpointSynapse,
    deserialize: bool,
    timeout: float,
    retries=1,
) -> List[TaskSynapse | SetOperatorEndpointSynapse | None] | None:
    res: List[TaskSynapse | SetOperatorEndpointSynapse | None] = [None] * len(axons)
    idx = list(range(len(axons)))
    axons = axons.copy()

    try:
        for attempt in range(retries):
            responses: List[TaskSynapse | SetOperatorEndpointSynapse] = await dendrite(
                axons=axons, synapse=synapse, deserialize=deserialize, timeout=timeout
            )

            new_idx = []
            new_axons = []
            for i, response in enumerate(responses):
                if (
                    response.dendrite.status_code is not None
                    and int(response.dendrite.status_code) == 422
                ):
                    if attempt == retries - 1:
                        res[idx[i]] = response
                        bt.logging.info(
                            "Wasn't able to get answers from axon {} after {} attempts".format(
                                axons[i], retries
                            )
                        )
                    else:
                        new_idx.append(idx[i])
                        new_axons.append(axons[i])
                else:
                    res[idx[i]] = response

            if len(new_idx):
                bt.logging.info(
                    "Found {} synapses with broken pipe, retrying them".format(
                        len(new_idx)
                    )
                )
            else:
                break

            idx = new_idx
            axons = new_axons

        assert all(el is not None for el in res)
        return res

    except Exception as e:
        bt.logging.error(f"Error while sending synapse with dendrite with retries {e}")

