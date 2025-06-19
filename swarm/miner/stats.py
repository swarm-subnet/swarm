from typing import Optional


class MinerStats:
    """
    A simple class for tracking aggregate statistics across multiple tasks:
      - number of tasks
      - average score
      - average execution time
    """

    def __init__(self):
        self.num_tasks: int = 0
        self.total_score: float = 0.0
        self.total_execution_time: float = 0.0

    def log_feedback(self, score: Optional[float], execution_time: Optional[float]):
        """
        Logs feedback by incrementing number of tasks and updating total
        score and total execution time.
        """
        if score is None:
            score = 0.0
        if execution_time is None:
            execution_time = 0.0

        self.num_tasks += 1
        self.total_score += score
        self.total_execution_time += execution_time

    @property
    def avg_score(self) -> float:
        if self.num_tasks == 0:
            return 0.0
        return self.total_score / self.num_tasks

    @property
    def avg_execution_time(self) -> float:
        if self.num_tasks == 0:
            return 0.0
        return self.total_execution_time / self.num_tasks
