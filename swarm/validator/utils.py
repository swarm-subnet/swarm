from swarm.protocol import TaskSynapse, SetOperatorEndpointSynapse
from swarm.utils.logging import ColoredLogger
import copy
import random
import bittensor as bt
from typing import List


def init_validator_performance_stats(validator) -> None:
    """
    Initialize a performance statistics dictionary on the validator if not present.
    This dictionary will track data across multiple forward calls.
    """
    if not hasattr(validator, "validator_performance_stats"):
        validator.validator_performance_stats = {
            "total_forwards_count": 0,  # how many forward passes occurred
            "total_forwards_time": 0.0,  # sum of all forward iteration times
            "total_tasks_generated": 0,  # how many tasks have been generated in total
            "total_generated_tasks_time": 0.0,  # total time spent generating tasks
            "total_processing_tasks_time": 0.0,  # total time spent in process_tasks
            "total_tasks_sent": 0,  # how many tasks have been sent overall (accum. from all forwards)
            "total_tasks_success": 0,  # tasks with at least one reward>0
            "total_tasks_wrong": 0,  # tasks with responses but no reward>0
            "total_tasks_no_response": 0,  # tasks with 0 valid responses
            "total_sum_of_avg_response_times": 0.0,  # sum of average miner solve times per task
            "total_sum_of_evaluation_times": 0.0,  # sum of times spent evaluating (score updates)
            "total_sum_of_avg_scores": 0.0,  # sum of average rewards per task
            "overall_tasks_processed": 0,  # total tasks processed for stats
        }


def update_validator_performance_stats(
    validator,
    tasks_count: int,
    num_success: int,
    num_wrong: int,
    num_no_response: int,
    sum_of_avg_response_times: float,
    sum_of_evaluation_times: float,
    sum_of_avg_scores: float,
) -> None:
    """
    Accumulates stats from a single batch of processed tasks into
    the validator's performance stats dictionary.
    """
    if not hasattr(validator, "validator_performance_stats"):
        init_validator_performance_stats(validator)

    vps = validator.validator_performance_stats

    # update global counters
    vps["total_tasks_sent"] += tasks_count
    vps["total_tasks_success"] += num_success
    vps["total_tasks_wrong"] += num_wrong
    vps["total_tasks_no_response"] += num_no_response

    # sums used to compute averages
    vps["total_sum_of_avg_response_times"] += sum_of_avg_response_times
    vps["total_sum_of_evaluation_times"] += sum_of_evaluation_times
    vps["total_sum_of_avg_scores"] += sum_of_avg_scores

    vps["overall_tasks_processed"] += tasks_count


def print_validator_performance_stats(validator) -> None:
    """
    Pretty-prints the validator performance stats using a Rich-styled table.
    """
    from rich.table import Table
    from rich.console import Console
    from rich import box

    vps = getattr(validator, "validator_performance_stats", None)
    if not vps:
        bt.logging.warning("No validator performance stats to display.")
        return

    # Compute derived stats
    total_forwards = vps["total_forwards_count"]
    avg_forward_time = (
        vps["total_forwards_time"] / total_forwards if total_forwards > 0 else 0.0
    )

    total_gen_tasks = vps["total_tasks_generated"]
    avg_task_gen_time = (
        vps["total_generated_tasks_time"] / total_gen_tasks
        if total_gen_tasks > 0
        else 0.0
    )

    overall_tasks = vps["overall_tasks_processed"]
    avg_processing_time_per_task = (
        vps["total_processing_tasks_time"] / overall_tasks if overall_tasks > 0 else 0.0
    )

    # success rate, etc
    tasks_sent = vps["total_tasks_sent"]
    tasks_success = vps["total_tasks_success"]
    tasks_wrong = vps["total_tasks_wrong"]
    tasks_no_resp = vps["total_tasks_no_response"]
    success_rate = (tasks_success / tasks_sent) if tasks_sent > 0 else 0.0

    avg_response_time = (
        vps["total_sum_of_avg_response_times"] / overall_tasks
        if overall_tasks > 0
        else 0.0
    )
    avg_evaluation_time = (
        vps["total_sum_of_evaluation_times"] / overall_tasks
        if overall_tasks > 0
        else 0.0
    )
    avg_score = (
        vps["total_sum_of_avg_scores"] / overall_tasks if overall_tasks > 0 else 0.0
    )

    console = Console()
    table = Table(
        title="[bold yellow]Validator Performance Stats[/bold yellow]",
        header_style="bold magenta",
        box=box.SIMPLE,
        expand=True,
    )
    table.add_column("Stat", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Forwards", str(total_forwards))
    table.add_row("Average Forward Time (s)", f"{avg_forward_time:.2f}")

    table.add_row("Tasks Generated (total)", str(total_gen_tasks))
    table.add_row(
        "Total Time Generating Tasks (s)", f"{vps['total_generated_tasks_time']:.2f}"
    )
    table.add_row("Average Time per Generated Task (s)", f"{avg_task_gen_time:.2f}")

    table.add_row("Tasks Processed (total)", str(tasks_sent))
    table.add_row("Successfull tasks", str(tasks_success))
    table.add_row("Not Successfull Tasks", str(tasks_wrong))
    table.add_row("Tasks with No Response", str(tasks_no_resp))
    table.add_row("Success Rate", f"{(success_rate * 100):.2f}%")

    table.add_row("Avg Miner Solve Time (s)", f"{avg_response_time:.2f}")
    table.add_row("Avg Evaluation Time per Task (s)", f"{avg_evaluation_time:.4f}")
    table.add_row("Avg Score per Task", f"{avg_score:.4f}")

    table.add_row(
        "Total Time Processing Tasks (s)", f"{vps['total_processing_tasks_time']:.2f}"
    )
    table.add_row(
        "Average Processing Time per Task (s)", f"{avg_processing_time_per_task:.2f}"
    )

    console.print(table)
    console.print()  # extra newline


# async def update_miner_stats_and_scores(
#     validator,
#     rewards: np.ndarray,
#     miner_uids: List[int],
#     execution_times: List[float],
#     task: Task,
# ) -> float:
#     """
#     Updates scores for miners based on computed rewards, updates local miner_stats,
#     and returns the time it took to evaluate miners.
#     """
#     evaluation_time = 0.0
#     if rewards is not None:
#         evaluation_time_start = time.time()
#         bt.logging.info("Scores updated for miners")

#         for i, miner_uid in enumerate(miner_uids):
#             miner_uid = int(miner_uid)
#             score_value = rewards[i] if rewards[i] is not None else 0.0
#             exec_time_value = (
#                 execution_times[i] if execution_times[i] is not None else TIMEOUT
#             )
#             success = score_value >= TIME_WEIGHT
#             if miner_uid not in validator.miner_stats:
#                 validator.miner_stats[miner_uid] = MinerStats()

#             validator.miner_stats[miner_uid].update(
#                 score=float(score_value),
#                 execution_time=float(exec_time_value),
#                 evaluation_time=(time.time() - evaluation_time_start),
#                 last_task=task,
#                 success=success,
#             )
#             validator.miner_stats["aggregated"].update(
#                 score=float(score_value),
#                 execution_time=float(exec_time_value),
#                 evaluation_time=(time.time() - evaluation_time_start),
#                 last_task=task,
#                 success=success,
#             )
#         evaluation_time_end = time.time()
#         evaluation_time = evaluation_time_end - evaluation_time_start
#     return evaluation_time


async def retrieve_random_demo_web_project() -> WebProject:
    """
    Retrieves a random demo web project from the available ones.
    Raises an Exception if none are available.
    """
    web_projects = await initialize_demo_webs_projects(demo_web_projects)
    bt.logging.debug(f"Retrieved {len(web_projects)} demo web projects.")
    if not web_projects:
        raise Exception("No demo web projects available.")
    project = random.choice(web_projects)
    ColoredLogger.info(
        f"Generating tasks for Web Project: '{project.name}'",
        ColoredLogger.YELLOW,
    )
    return project


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


def prepare_for_feedback(task) -> Task:
    cleaned_task = copy.deepcopy(task)
    cleaned_task.use_case = None
    cleaned_task.milestones = None
    cleaned_task.interactive_elements = None
    cleaned_task.screenshot = None
    cleaned_task.screenshot_description = None
    cleaned_task.html = None
    cleaned_task.clean_html = None

    return cleaned_task
