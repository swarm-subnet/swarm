import bittensor as bt
from typing import List
from swarm.protocol import TaskSynapse


async def dendrite_with_retries(
    dendrite: bt.dendrite,
    axons: list,
    synapse: TaskSynapse,
    deserialize: bool,
    timeout: float,
    cnt_attempts=3
) -> List[TaskSynapse]:
    res: List[TaskSynapse | None] = [None] * len(axons)
    idx = list(range(len(axons)))
    axons = axons.copy()

    for attempt in range(cnt_attempts):
        responses: List[TaskSynapse] = await dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout
        )

        new_idx = []
        new_axons = []
        for i, response in enumerate(responses):
            if response.dendrite.status_code is not None and int(response.dendrite.status_code) == 422:
                if attempt == cnt_attempts - 1:
                    res[idx[i]] = response
                    bt.logging.info(
                        "Wasn't able to get answers from axon {} after {} attempts".format(axons[i], cnt_attempts)
                    )
                else:
                    new_idx.append(idx[i])
                    new_axons.append(axons[i])
            else:
                res[idx[i]] = response

        if len(new_idx):
            bt.logging.info(
                'Found {} synapses with broken pipe, retrying them'.format(len(new_idx))
            )
        else:
            break

        idx = new_idx
        axons = new_axons

    assert all(el is not None for el in res)
    return res
