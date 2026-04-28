"""Placeholder training environment wrapper.

Intended role:
- normalize how training code creates tasks
- expose any training-only labels in one place
- keep the deploy-time observation contract unchanged for the actor
"""


def describe_training_env() -> str:
    return (
        "Placeholder training environment wrapper. "
        "Fill in task creation, eval splits, and optional training-only labels."
    )
